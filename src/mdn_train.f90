program mdn_train
  use mdn
  use mlp
  use mod_foptparse
  implicit none

  type program_settings
    integer::n_committee_members
    integer::n_hidden_min
    integer::n_hidden_max
    integer::n_kern_min
    integer::n_kern_max
    integer::max_training_iter
    character(len=OPT_LEN_CHARVAL)::committee_output_file
    character(len=OPT_LEN_CHARVAL)::training_data_file
    character(len=OPT_LEN_CHARVAL)::monitor_data_file
  end type program_settings
  type(program_settings)::settings
  type(mlp_settings),dimension(:),allocatable::mlp_structure
  type(mdn_committee)::committee
  real(k_rd),dimension(:,:),allocatable::inputs,monitor_inputs
  real(k_rd),dimension(:),allocatable::targets,monitor_targets
  integer::ict,ieg
  real(k_rd),dimension(:),allocatable::biases

  call init_random_seed()
  call get_program_settings(settings)

  call load_dataset(settings%training_data_file,inputs,targets)
  call load_dataset(settings%monitor_data_file,monitor_inputs,monitor_targets)
  if(size(inputs,1).ne.size(monitor_inputs,1)) stop "Training and monitoring sets appear to have different dimensions"

  call set_committee_structure(settings%n_committee_members,size(inputs,1),1, &
                                settings%n_hidden_min,settings%n_hidden_max,&
                                settings%n_kern_min,settings%n_kern_max,&
                                mlp_structure)

  call create_mdn_committee(committee,settings%n_committee_members,size(inputs,1),mlp_structure)
  committee%standardise_mean(1:size(inputs,1)) = sum(inputs,2)/real(size(inputs,2),k_rd)
  do ieg=1,size(inputs,2)
    inputs(:,ieg) = inputs(:,ieg)-committee%standardise_mean(1:size(inputs,1))
  end do
  committee%standardise_std(1:size(inputs,1)) = (sum(inputs**2.,2)/real(size(inputs,2)-1,k_rd))**0.5
  do ieg=1,size(inputs,2)
    inputs(:,ieg) = inputs(:,ieg)/committee%standardise_std(1:size(inputs,1))
  end do
  committee%standardise_mean(size(inputs,1)+1)=sum(targets)/real(size(targets,1),k_rd)
  targets=targets-committee%standardise_mean(size(inputs,1)+1)
  committee%standardise_std(size(inputs,1)+1) = (sum(targets**2.)/real(size(targets,1)-1,k_rd))**0.5
  targets = targets/committee%standardise_std(size(inputs,1)+1)
  do ieg=1,size(monitor_inputs,2)
    monitor_inputs(:,ieg) = standardise(monitor_inputs(:,ieg),committee%standardise_mean(1:size(inputs,1)), &
                                          committee%standardise_std(1:size(inputs,1)))
  end do
  monitor_targets = standardise(monitor_targets,committee%standardise_mean(size(inputs,1)+1), &
                                          committee%standardise_std(size(inputs,1)+1))

  do ict=1,settings%n_committee_members
    allocate(biases(3*committee%nkernels(ict)))
    call evaluate_initial_biases(targets,biases)
    call initialise_mlp(committee%member(ict),biases(1:3*committee%nkernels(ict)))
    deallocate(biases)
  end do
  call train_mdn_committee(committee,inputs,targets,0.0_k_rd,settings%max_training_iter,monitor_inputs,monitor_targets)
  call store_mdn_committee(committee,settings%committee_output_file)

contains
  subroutine set_committee_structure(nct,ninputs,noutputs,hidmin,hidmax,kernmin,kernmax,settings)
    integer,intent(in)::nct,ninputs,noutputs,hidmin,hidmax,kernmin,kernmax
    real(k_rd)::r
    type(mlp_settings),dimension(:),allocatable,intent(out)::settings
    integer::ict
    allocate(settings(nct))
    do ict=1,nct
      settings(ict)%nhidden = 1
      allocate(settings(ict)%node_counts(3),settings(ict)%neuron_type(2))
      settings(ict)%node_counts(1)=ninputs
      if(hidmax.lt.hidmin) then
        stop "set_committee_structure: Nonsensical inputs"
      else if (hidmax.eq.hidmin) then
        settings(ict)%node_counts(2)=hidmin
      else
        call random_number(r)
        settings(ict)%node_counts(2)=hidmin+int(floor(real(hidmax+1-hidmin,k_rd)*r))
      end if
      if(kernmax.lt.kernmin) then
        stop "set_committee_structure: Nonsensical inputs"
      else if(kernmin.eq.kernmax) then
        settings(ict)%node_counts(3)=3*kernmin
      else
        call random_number(r)
        settings(ict)%node_counts(3)=3*(kernmin+int(floor(real(kernmax+1-kernmin,k_rd)*r)))
      end if
      settings(ict)%neuron_type(1)=NEURON_TYPE_TANH
      settings(ict)%neuron_type(2)=NEURON_TYPE_LINEAR
    end do
  end subroutine set_committee_structure

  subroutine get_program_settings(settings)
    type(program_settings),intent(inout)::settings
    integer,parameter::nopts=8
    type(fopt_opt),dimension(nopts)::option
    character(len=OPT_LEN_CHARVAL),dimension(:),allocatable::arg
    character(len=OPT_LEN_HELPTEXT)::helpheader,helpfooter
    integer::nargs
    integer,parameter::OPT_OUTPUT_FILE=7
    integer,parameter::OPT_NUM_CT_MEMBERS=1
    integer,parameter::OPT_N_HIDDEN_MIN=2,OPT_N_HIDDEN_MAX=3
    integer,parameter::OPT_N_KERN_MIN=4,OPT_N_KERN_MAX=5
    integer,parameter::OPT_MAX_TRAINING_ITER=6
    integer,parameter::OPT_HELP=nopts

    call fopt_setup_option(option(OPT_OUTPUT_FILE),'output-file',ARG_CHAR,shortform='o',charval='mdnc.out',&
                            helptext="Output file for trained MDN committee",metavar="FILE")
    call fopt_setup_option(option(OPT_NUM_CT_MEMBERS),'n-ct-members',ARG_INT,shortform='n',intval=10,&
                            helptext="Number of individual MDNs within committee",metavar="N")
    call fopt_setup_option(option(OPT_N_HIDDEN_MIN),'n-hid-min',ARG_INT,shortform='l',intval=25,&
                            helptext="Minimum number of nodes in hidden layer")
    call fopt_setup_option(option(OPT_N_HIDDEN_MAX),'n-hid-max',ARG_INT,shortform='u',intval=50,&
                            helptext="Maximum number of nodes in hidden layer")
    call fopt_setup_option(option(OPT_N_KERN_MIN),'n-kern-min',ARG_INT,shortform='j',intval=3,&
                            helptext="Minimum number of Gaussian kernels in MDN")
    call fopt_setup_option(option(OPT_N_KERN_MAX),'n-kern-max',ARG_INT,shortform='k',intval=8,&
                            helptext="Maximum number of Gaussian kernels in MDN")
    call fopt_setup_option(option(OPT_MAX_TRAINING_ITER),'max-iter',ARG_INT,shortform='t',intval=1000,&
                            helptext="Maximum number of iterations to perform during training")
    call fopt_setup_option(option(OPT_HELP),'help',ARG_HELP,shortform='h',helptext="Print this help text")
    helpheader = 'Usage: mdn_train [options] training_data monitor_data'
    helpfooter = 'Program to create and train an MDN committee.'
    call fopt_parse(nopts,option,arg,nargs,6,helpheader,helpfooter)
    if(nargs.ne.2) then
      call fopt_write_help(6,helpheader,helpfooter,nopts,option)
      stop
    end if
    settings%n_committee_members = option(OPT_NUM_CT_MEMBERS)%intval
    settings%n_hidden_min = option(OPT_N_HIDDEN_MIN)%intval
    settings%n_hidden_max = option(OPT_N_HIDDEN_MAX)%intval
    settings%n_kern_min = option(OPT_N_KERN_MIN)%intval
    settings%n_kern_max = option(OPT_N_KERN_MAX)%intval
    settings%max_training_iter = option(OPT_MAX_TRAINING_ITER)%intval
    settings%committee_output_file = option(OPT_OUTPUT_FILE)%charval
    settings%training_data_file = arg(1)
    settings%monitor_data_file = arg(2)
    deallocate(arg)
  end subroutine get_program_settings

end program mdn_train
