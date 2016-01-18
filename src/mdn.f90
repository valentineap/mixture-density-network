module mdn
  use mlp
  implicit none

contains
  subroutine evaluate_gmm(nx,x,p,nkernels,weights,means,stds)
    integer,intent(in)::nx ! Number of evaluation points
    real(k_rd),dimension(:),intent(in)::x
    real(k_rd),dimension(:),intent(out)::p
    integer,intent(in)::nkernels
    real(k_rd),dimension(:),intent(in)::weights,means,stds
    integer::i,j
    real(k_rd)::pi=4.0_k_rd*atan(1.0_k_rd)
    if( (size(x,1).lt.nx).or. (size(p,1).lt.nx) .or. (size(weights,1).lt.nkernels) .or. &
                        (size(means,1).lt.nkernels) .or. (size(stds,1).lt.nkernels)) stop 'Dimension error 1'
    do i=1,nx
      p(i)=0.0_k_rd
      do j=1,nkernels
        p(i) = p(i) + weights(j)*exp(-((x(i)-means(j))**2.)/(2.0_k_rd*stds(j)**2.))/(stds(j)*(2*pi)**0.5_k_rd)
      end do
    end do
  end subroutine evaluate_gmm

  subroutine evaluate_gmm_from_network_outputs(nx,x,p,nkernels,netouts)
    integer,intent(in)::nx ! Number of evaluation points
    real(k_rd),dimension(:),intent(in)::x
    real(k_rd),dimension(:),intent(out)::p
    integer,intent(in)::nkernels
    real(k_rd),dimension(:),intent(in),target::netouts
    real(k_rd),pointer,dimension(:)::p2wts,p2means,p2stds
    if( (size(x,1).lt.nx).or. (size(p,1).lt.nx) .or. (size(netouts).lt.3*nkernels) ) stop 'Dimension error 2'
    p2wts => netouts(1:3*nkernels-2:3)
    p2means => netouts(2:3*nkernels-1:3)
    p2stds => netouts(3:3*nkernels:3)
    call evaluate_gmm(nx,x,p,nkernels,exp(p2wts)/sum(exp(p2wts)),p2means,exp(p2stds))
  end subroutine

  subroutine evaluate_gaussian_ratio(nx,x,nkernels,weights,means,stds,ratio)
    integer,intent(in)::nx ! Number of evaluation points
    real(k_rd),dimension(:),intent(in)::x
    integer,intent(in)::nkernels
    real(k_rd),dimension(:),intent(in)::weights,means,stds
    real(k_rd),dimension(:,:),intent(out)::ratio
    integer::i,j,k
    if( (size(x,1).lt.nx) .or. (size(weights,1).lt.nkernels) .or. (size(means,1).lt.nkernels) .or. &
                  (size(stds,1).lt.nkernels) .or. (size(ratio,2).lt.nx) .or. size(ratio,1).lt.nkernels ) stop 'Dimension error 3'
    do i = 1,nx
      do j=1,nkernels
        ratio(j,i) = 1.0_k_rd/sum(weights(:)*stds(j)* exp( ( (stds(:)*(x(i)-means(j)))**2. - &
                          (stds(j)*(x(i)-means(:)))**2. ) / (2.0_k_rd*(stds(:)*stds(j))**2.)) / stds(:))
      end do
    end do
  end subroutine evaluate_gaussian_ratio

  subroutine evaluate_error_and_derivatives(nx,x,nkernels,netouts,error,deriv)
    integer,intent(in)::nx ! Number of evaluation points
    real(k_rd),dimension(:),intent(in)::x
    integer,intent(in)::nkernels
    real(k_rd),dimension(:),intent(in)::netouts
    real(k_rd),intent(out),dimension(:)::error
    real(k_rd),dimension(:,:),intent(out)::deriv
    real(k_rd),dimension(nkernels)::wts,means,stds
    real(k_rd),dimension(nx)::p
    real(k_rd),dimension(nkernels,nx)::ratio
    integer::ikern
    if( (size(x,1).lt.nx).or. (size(netouts).lt.3*nkernels) .or. (size(error,1) .lt. nx) .or. &
                      (size(deriv,1).lt.3*nkernels) .or. (size(deriv,2).lt.nx ) ) stop 'Dimension error 4'
    wts = netouts(1:3*nkernels-2:3)
    wts = exp(wts)/sum(exp(wts))
    means = netouts(2:3*nkernels-1:3)
    stds = netouts(3:3*nkernels:3)
    stds = exp(stds)
    call evaluate_gmm(nx,x,p,nkernels,wts,means,stds)
    error = -log(p)
    call evaluate_gaussian_ratio(nx,x,nkernels,wts,means,stds,ratio)
    do ikern = 1,nkernels
      deriv( 3*(ikern-1) + 1,:) = - wts(ikern)*(ratio(ikern,:)-1.0_k_rd)
      deriv( 3*(ikern-1) + 2,:) = - wts(ikern)*(x-means(ikern))*ratio(ikern,:)/stds(ikern)**2.
      deriv( 3*(ikern-1) + 3,:) = - wts(ikern)*(((x-means(ikern))/stds(ikern))**2. -1.0_k_rd )*ratio(ikern,:)
    end do
  end subroutine evaluate_error_and_derivatives


  subroutine evaluate_mdn(nkernels,mlp,neg,inputs,nx,x,p)
    integer,intent(in)::nkernels
    type(mlp_network),intent(in)::mlp
    integer,intent(in)::neg
    real(k_rd),dimension(:,:),intent(in)::inputs !inputs(:,neg)
    integer,intent(in)::nx
    real(k_rd),dimension(:),intent(in)::x !x(nx)
    real(k_rd),dimension(:,:),intent(out)::p !p(nx,neg)
    real(k_rd),dimension(3*nkernels,neg)::netouts
    integer::ieg
    if(3*nkernels .ne. get_layer_nout(mlp,mlp%nlayers)) stop 'Wrong mlp?'

    call evaluate_mlp(mlp,inputs,netouts)
    do ieg = 1,neg
      call evaluate_gmm_from_network_outputs(nx,x,p(:,ieg),nkernels,netouts(:,ieg))
    end do
  end subroutine evaluate_mdn

  subroutine evaluate_mdn_error_and_derivatives(nkernels,mlp,neg,inputs,targets,E,dEdp)
    integer,intent(in)::nkernels
    type(mlp_network),intent(in)::mlp
    integer,intent(in)::neg
    real(k_rd),dimension(:,:),intent(in)::inputs !inputs(:,neg)
    real(k_rd),dimension(:),intent(in)::targets !targets(neg)
    real(k_rd),intent(out)::E
    real(k_rd),intent(out),dimension(:),optional::dEdp !dEdp(mlp%nparams)
    real(k_rd),dimension(:,:),allocatable::mlp_internal,mlp_internal_deriv
    real(k_rd),dimension(3*nkernels,neg)::outputs
    real(k_rd),dimension(3*nkernels,neg),target::dEk_dnetout
    real(k_rd),dimension(1,neg)::Ek
    integer::i1,i2,ilayer,n_internal,ieg
    real(k_rd),pointer,dimension(:,:)::p2dEdx,p2tmp

    n_internal = 0
    do ilayer = 1,mlp%nlayers
      n_internal = n_internal + get_layer_nout(mlp,ilayer)
    end do
    allocate(mlp_internal(n_internal,neg),mlp_internal_deriv(n_internal,neg))

    call evaluate_mlp(mlp,inputs,outputs,mlp_internal,mlp_internal_deriv)
    do ieg = 1,neg
      call evaluate_error_and_derivatives(1,targets(ieg:ieg),nkernels,outputs(:,ieg),Ek(:,ieg),dEk_dnetout(:,ieg:ieg))
    end do
    E = sum(Ek)
    if(present(dEdp)) then
      if(size(dEdp,1).lt.mlp%nparams) stop 'dimension error 5'
      i1 = n_internal+1
      allocate(p2dEdx(3*nkernels,neg))
      p2dEdx = dEk_dnetout
      do ilayer = mlp%nlayers,2,-1
        i2 = i1-1
        i1 = i2 - get_layer_nout(mlp,ilayer)+1
        dEdp(mlp%ib(1,ilayer):mlp%ib(2,ilayer)) = sum(p2dEdx*mlp_internal_deriv(i1:i2,:),2)
        dEdp(mlp%iw(1,ilayer):mlp%iw(2,ilayer)) = reshape(matmul(p2dEdx*mlp_internal_deriv(i1:i2,:), &
                                                    transpose(mlp_internal(i1-get_layer_nout(mlp,ilayer-1):i1-1,:))), &
                                                    (/mlp%iw(2,ilayer)-mlp%iw(1,ilayer)+1/))
        allocate(p2tmp(get_layer_nout(mlp,ilayer-1),neg))
        p2tmp = matmul(transpose(reshape(mlp%params(mlp%iw(1,ilayer):mlp%iw(2,ilayer)), &
                                                    (/get_layer_nout(mlp,ilayer),get_layer_nin(mlp,ilayer)/))) &
                                            ,p2dEdx*mlp_internal_deriv(i1:i2,:))
        deallocate(p2dEdx)
        p2dEdx =>p2tmp
        nullify(p2tmp)
      end do
      i2=i1-1
      i1 = i2 - get_layer_nout(mlp,1)+1
      dEdp(mlp%ib(1,1):mlp%ib(2,1)) = sum(p2dEdx*mlp_internal_deriv(i1:i2,:),2)
      dEdp(mlp%iw(1,1):mlp%iw(2,1)) = reshape(matmul(p2dEdx*mlp_internal_deriv(i1:i2,:),transpose(inputs)), &
                                              (/mlp%iw(2,1)-mlp%iw(1,1)+1/))
      deallocate(p2dEdx)
      dEdp(mlp%iw(1,mlp%nlayers):mlp%iw(2,mlp%nlayers))=0.0_k_rd
      dEdp(mlp%ib(1,mlp%nlayers):mlp%ib(2,mlp%nlayers))=0.0_k_rd
    end if

    deallocate(mlp_internal,mlp_internal_deriv)
  end subroutine evaluate_mdn_error_and_derivatives



  subroutine train_mdn(nkernels,mlp,neg,inputs,targets,noise_std,maxit,monitor_inputs,monitor_targets,best_params,best_E)
    integer,intent(in)::nkernels
    type(mlp_network),intent(inout)::mlp
    integer,intent(in)::neg
    real(k_rd),dimension(:,:),intent(in)::inputs
    real(k_rd),dimension(:),intent(in)::targets
    real(k_rd),dimension(:,:),allocatable::noise
    real(k_rd),intent(in)::noise_std
    integer,intent(in)::maxit
    real(k_rd),intent(in),dimension(:,:),optional::monitor_inputs
    real(k_rd),intent(in),dimension(:),optional::monitor_targets
    real(k_rd),intent(inout),dimension(:),optional::best_params
    real(k_rd),intent(inout),optional::best_E
    integer,  parameter    :: m = 10, iprint = 50
    real(k_rd), parameter    :: factr = 1.0_k_rd,pgtol=0.0_k_rd!factr  = 1.0d+7, pgtol  = 1.0d-5
    character(len=60)      :: task, csave
    logical                :: lsave(4)
    integer                :: isave(44)
    real(k_rd)               :: E,Emon
    real(k_rd)               :: dsave(29)
    integer,  allocatable  :: nbound(:), iwa(:)
    real(k_rd), allocatable  :: lower(:), upper(:), grad(:), wa(:)
    integer::i,j,nmon,iminibatch
    allocate ( nbound(mlp%nparams), grad(mlp%nparams), noise(size(inputs,1),size(inputs,2)))
    allocate ( iwa(3*mlp%nparams) )
    allocate ( wa(2*m*mlp%nparams + 5*mlp%nparams + 11*m*m + 8*m) )
    nbound = 0




    call evaluate_mdn_error_and_derivatives(nkernels,mlp,neg,inputs,targets,E,grad)
    if(present(monitor_inputs)) then
      if(.not.((present(monitor_targets) .and. present(best_params) .and. present(best_E)))) stop 'Monitor error'
      nmon = size(monitor_inputs,2)
    end if

    !print *, E
    !do iminibatch=1,10
    task = 'START'
    do j=1,size(inputs,2)
      do i=1,size(inputs,1)
        noise(i,j)=noise_std*rnor()
      end do
    end do
    do while( (task(1:2).eq.'FG') .or. (task(1:5).eq.'START') .or. (task(1:5).eq.'NEW_X') )
      call setulb(mlp%nparams,m,mlp%params,lower,upper,nbound,E,grad,factr,pgtol, &
                  wa,iwa,task,iprint,csave,lsave,isave,dsave)
      if(task(1:2).eq.'FG') then
        if(isave(30) .gt. maxit) then
          task = 'STOP: MAXIMUM ITERATIONS'
        else
          call evaluate_mdn_error_and_derivatives(nkernels,mlp,neg,inputs+noise,targets,E,grad)
          if(present(monitor_inputs)) then
            call evaluate_mdn_error_and_derivatives(nkernels,mlp,nmon,monitor_inputs,monitor_targets,Emon)
            if(Emon .lt. best_E) then
              best_params = mlp%params
              best_E = Emon
              print *, Emon
            end if
          end if
        end if
      end if
    end do
    !end do
  end subroutine train_mdn

  subroutine evaluate_initial_biases(nkern,neg,targets,biases)
    integer,intent(in)::nkern,neg
    real(k_rd),dimension(:)::targets
    integer::it_num
    integer,dimension(neg)::cluster
    real(k_rd),dimension(1,nkern)::cluster_center
    integer,dimension(nkern)::cluster_population
    real(k_rd),dimension(nkern)::cluster_energy
    real(k_rd),dimension(3*nkern)::biases
    integer::i

    cluster_center(1,1)=minval(targets)
    cluster_center(1,nkern) = maxval(targets)
    do i=2,nkern-1
      cluster_center(1,i) = cluster_center(1,1)+real(i-1,k_rd)*(cluster_center(1,nkern)-cluster_center(1,1))/real(nkern-1,k_rd)
    end do
    print *, cluster_center
    call kmeans_02 (1,neg,nkern,10000, it_num, reshape(targets,(/1,neg/)), &
      cluster, cluster_center, cluster_population, cluster_energy )
    print *, cluster_center
    print *, cluster_population

    do i = 1,nkern
      biases(3*(i-1)+1) = log(real(cluster_population(i),k_rd))
      biases(3*(i-1)+2) = cluster_center(1,i)
      biases(3*(i-1)+3) = log(sqrt(cluster_energy(i)/real(cluster_population(i))))
    end do
    print *, biases
  end subroutine


  !From GCC website
  subroutine init_random_seed()
            use iso_fortran_env, only: int64
            implicit none
            integer, allocatable :: seed(:)
            integer :: i, n, un, istat, dt(8), pid
            integer(int64) :: t

            call random_seed(size = n)
            allocate(seed(n))
            ! First try if the OS provides a random number generator
            open(newunit=un, file="/dev/urandom", access="stream", &
                 form="unformatted", action="read", status="old", iostat=istat)
            if (istat == 0) then
               read(un) seed
               close(un)
            else
               ! Fallback to XOR:ing the current time and pid. The PID is
               ! useful in case one launches multiple instances of the same
               ! program in parallel.
               call system_clock(t)
               if (t == 0) then
                  call date_and_time(values=dt)
                  t = (dt(1) - 1970) * 365_int64 * 24 * 60 * 60 * 1000 &
                       + dt(2) * 31_int64 * 24 * 60 * 60 * 1000 &
                       + dt(3) * 24_int64 * 60 * 60 * 1000 &
                       + dt(5) * 60 * 60 * 1000 &
                       + dt(6) * 60 * 1000 + dt(7) * 1000 &
                       + dt(8)
               end if
               pid = getpid()
               t = ieor(t, int(pid, kind(t)))
               do i = 1, n
                  seed(i) = lcg(t)
               end do
            end if
            call random_seed(put=seed)
          contains
            ! This simple PRNG might not be good enough for real work, but is
            ! sufficient for seeding a better PRNG.
            function lcg(s)
              integer :: lcg
              integer(int64) :: s
              if (s == 0) then
                 s = 104729
              else
                 s = mod(s, 4294967296_int64)
              end if
              s = mod(s * 279470273_int64, 4294967291_int64)
              lcg = int(mod(s, int(huge(0), int64)), kind(0))
            end function lcg
          end subroutine init_random_seed
end module mdn
