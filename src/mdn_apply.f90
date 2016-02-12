program mdn_apply
  use mdn
  use mod_foptparse
  implicit none

  type program_settings
    character(len=OPT_LEN_CHARVAL)::mdn_file
    character(len=OPT_LEN_CHARVAL)::data_file
    logical::input_has_targets
    logical::output_pdf
    real(k_rd)::pdf_lo,pdf_up
    integer::pdf_np
    character(len=OPT_LEN_CHARVAL)::output_file
    logical::copy_inputs
    logical::copy_targets
  end type program_settings
  type(program_settings)::settings
  type(mdn_committee)::ct
  real(k_rd),dimension(:,:),allocatable::inputs,weights,means,stds,p
  real(k_rd),dimension(:),allocatable::targets,x
  integer::ieg,ik,i,lu

  call get_program_settings(settings)
  call load_mdn_committee(ct,settings%mdn_file)

  ! Read in inputs and apply standardisation transformations
  if(settings%input_has_targets) then
    call load_dataset(settings%data_file,inputs,targets)
    targets=standardise(targets,ct%standardise_mean(ct%ninputs+1),ct%standardise_std(ct%ninputs+1))
  else
    call load_dataset(settings%data_file,inputs)
  end if
  do ieg=1,size(inputs,2)
    inputs(:,ieg) = standardise(inputs(:,ieg),ct%standardise_mean(1:ct%ninputs),ct%standardise_std(1:ct%ninputs))
  end do


  if(trim(settings%output_file).eq."") then
    lu = 6
  else
    open(newunit=lu,file=settings%output_file)
  end if

  if(settings%output_pdf) then
    allocate(x(settings%pdf_np),p(settings%pdf_np,size(inputs,2)))
    settings%pdf_lo = standardise(settings%pdf_lo,ct%standardise_mean(ct%ninputs+1),ct%standardise_std(ct%ninputs+1))
    settings%pdf_up = standardise(settings%pdf_up,ct%standardise_mean(ct%ninputs+1),ct%standardise_std(ct%ninputs+1))
    do i =1,settings%pdf_np
      x(i)=settings%pdf_lo + (i-1)*(settings%pdf_up-settings%pdf_lo)/real(settings%pdf_np-1,k_rd)
    end do
    p = evaluate_mdn_committee(ct,inputs,x)
    if(settings%copy_inputs) then
      if(settings%copy_targets) then
        do ieg=1,size(inputs,2)
          write(lu,*) (unstandardise(inputs(i,ieg),ct%standardise_mean(i),ct%standardise_std(i)),i=1,size(inputs,1)) &
                      ,unstandardise(targets(ieg),ct%standardise_mean(size(inputs,1)+1), &
                        ct%standardise_std(ct%ninputs+1)),(p(i,ieg)/ct%standardise_std(size(inputs,1)+1),i=1,settings%pdf_np)
        end do
      else
        do ieg=1,size(inputs,2)
          write(lu,*) (unstandardise(inputs(i,ieg),ct%standardise_mean(i),ct%standardise_std(i)),i=1,size(inputs,1)), &
                          (p(i,ieg)/ct%standardise_std(ct%ninputs+1),i=1,settings%pdf_np)
        end do
      end if
    else
      if(settings%copy_targets) then
        do ieg=1,size(inputs,2)
          write(lu,*) unstandardise(targets(ieg),ct%standardise_mean(ct%ninputs+1),ct%standardise_std(size(inputs,1)+1)), &
                                      (p(i,ieg)/ct%standardise_std(ct%ninputs+1),i=1,settings%pdf_np)
        end do
      else
        do ieg=1,size(inputs,2)
          write(lu,*) (p(i,ieg)/ct%standardise_std(size(inputs,1)+1),i=1,settings%pdf_np)
        end do
      end if
    end if
    deallocate(x,p)
  else
    call evaluate_mdn_committee_to_kernel_parameters(ct,inputs,weights,means,stds)
    do ieg = 1,size(inputs,2)
      write(lu,*) ((/weights(ik,ieg)/ct%standardise_std(ct%ninputs+1),unstandardise(means(ik,ieg), &
                              ct%standardise_mean(ct%ninputs+1),ct%standardise_std(ct%ninputs+1)), &
                              ct%standardise_std(ct%ninputs+1)*stds(ik,ieg)/),ik=1,sum(ct%nkernels))
    end do
    deallocate(weights,means,stds)
  end if
  if(lu.ne.6) close(lu)
contains
  subroutine get_program_settings(settings)
    type(program_settings),intent(inout)::settings
    integer,parameter::nopts=9
    type(fopt_opt),dimension(nopts)::option
    character(len=OPT_LEN_CHARVAL),dimension(:),allocatable::arg
    character(len=OPT_LEN_HELPTEXT)::helpheader,helpfooter
    integer::nargs
    integer,parameter::OPT_HELP=nopts,OPT_HAS_TARGETS=1,OPT_OUTPUT_PDF=2,OPT_PDF_LO=3,OPT_PDF_UP=4,OPT_PDF_NP=5
    integer,parameter::OPT_COPY_INPUTS=7,OPT_COPY_TARGETS=8
    integer,parameter::OPT_OUTPUT_FILE=6
    call fopt_setup_option(option(OPT_HAS_TARGETS),'has-targets',ARG_NONE,switchval=.false., &
                            helptext="Datafile contains targets as well as inputs")
    call fopt_setup_option(option(OPT_OUTPUT_PDF),'output-pdf',ARG_NONE,shortform='p',switchval=.false., &
                            helptext="Output p(m|d). See also --m-lo, --m-up, --m-num")
    call fopt_setup_option(option(OPT_PDF_LO),'m-lo',ARG_FLOAT,floatval=-1.0,helptext="Minimum value for pdf evaluation")
    call fopt_setup_option(option(OPT_PDF_UP),'m-up',ARG_FLOAT,floatval=1.0,helptext="Maximum value for pdf evaluation")
    call fopt_setup_option(option(OPT_PDF_NP),'m-num',ARG_INT,intval=100,helptext="Number of points for pdf evaluation")
    call fopt_setup_option(option(OPT_OUTPUT_FILE),'output-file',ARG_CHAR,shortform='o',charval="",helptext="Filename for output")
    call fopt_setup_option(option(OPT_COPY_INPUTS),'copy-inputs',ARG_NONE,switchval=.false., &
                            helptext="Duplicate input values into output file")
    call fopt_setup_option(option(OPT_COPY_TARGETS),'copy-targets',ARG_NONE,switchval=.false.,&
                            helptext="Duplicate target values into output file")
    call fopt_setup_option(option(OPT_HELP),'help',ARG_HELP,shortform='h',helptext="Print this help text")
    helpheader = 'Usage: mdn_apply [options] mdn_file datafile'
    helpfooter = 'Program to create and train an MDN ct.'
    call fopt_parse(nopts,option,arg,nargs,6,helpheader,helpfooter)
    if(nargs.ne.2) then
      call fopt_write_help(6,helpheader,helpfooter,nopts,option)
      stop
    end if
    settings%mdn_file = arg(1)
    settings%data_file = arg(2)
    settings%input_has_targets = option(OPT_HAS_TARGETS)%switchval
    settings%output_pdf = option(OPT_OUTPUT_PDF)%switchval
    settings%pdf_lo = option(OPT_PDF_LO)%floatval
    settings%pdf_up = option(OPT_PDF_UP)%floatval
    settings%pdf_np = option(OPT_PDF_NP)%intval
    settings%output_file = option(OPT_OUTPUT_FILE)%charval
    settings%copy_inputs=option(OPT_COPY_INPUTS)%switchval
    settings%copy_targets = option(OPT_COPY_TARGETS)%switchval
    if(settings%copy_targets.and..not.settings%input_has_targets) then
      print *,
      print *,"!!! Warning: --copy-targets without --input-has-targets; ignored."
      print *,
      settings%copy_targets=.false.
    end if
    deallocate(arg)
  end subroutine get_program_settings
end program mdn_apply
