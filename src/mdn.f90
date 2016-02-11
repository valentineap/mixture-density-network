module mdn
  use mlp
  use iso_fortran_env, only: int64
  implicit none

  type mdn_committee
    integer::nmembers
    integer,dimension(:),allocatable::nkernels
    type(mlp_network),dimension(:),allocatable::member
    real(k_rd),dimension(:),allocatable::weight
  end type
  real(k_rd)::pi=4.0_k_rd*atan(1.0_k_rd)
contains
  elemental function gauss(x,mu,sigma) result(g)
    real(k_rd),intent(in)::x,mu,sigma
    real(k_rd)::g
    g = exp((-(x-mu)**2.0_k_rd)/(2.0_k_rd*sigma**2.0_k_rd))/(sigma*(2.0_k_rd*pi)**0.5_k_rd)
  end function gauss

  function evaluate_gmm(x,weights,means,stds) result(p)
    real(k_rd),dimension(:),intent(in)::x
    real(k_rd),dimension(:),intent(in)::weights
    real(k_rd),dimension(size(weights,1)),intent(in)::means,stds
    real(k_rd),dimension(size(x,1))::p
    integer::i
    p=0.0_k_rd
    do i=1,size(weights,1)
      p = p + weights(i)*gauss(x,means(i),stds(i))
    end do
  end function evaluate_gmm

  function evaluate_gmm_from_network_outputs(x,netouts) result(p)
    real(k_rd),dimension(:),intent(in)::x
    real(k_rd),dimension(size(x,1))::p
    real(k_rd),dimension(:),intent(in),target::netouts
    real(k_rd),pointer,dimension(:)::p2wts,p2means,p2stds
    integer::nkernels
    nkernels = size(netouts,1)/3
    if(3 * nkernels .ne. size(netouts,1)) stop 'Unexpected dimension for netouts'
    p2wts => netouts(1:3*nkernels-2:3)
    p2means => netouts(2:3*nkernels-1:3)
    p2stds => netouts(3:3*nkernels:3)
    p = evaluate_gmm(x,exp(p2wts)/sum(exp(p2wts)),p2means,exp(p2stds))
  end function

  subroutine evaluate_gaussian_ratio(x,weights,means,stds,ratio)
    real(k_rd),dimension(:),intent(in)::x
    real(k_rd),dimension(:),intent(in)::weights
    real(k_rd),dimension(size(weights,1))::means,stds
    real(k_rd),dimension(size(weights,1),size(x,1)),intent(out)::ratio
    integer::i,j,k
    do i = 1,size(x,1)
      do j=1,size(weights,1)
        ratio(j,i) = 1.0_k_rd/sum(weights(:)*stds(j)* exp( ( (stds(:)*(x(i)-means(j)))**2. - &
                          (stds(j)*(x(i)-means(:)))**2. ) / (2.0_k_rd*(stds(:)*stds(j))**2.)) / stds(:))
      end do
    end do
  end subroutine evaluate_gaussian_ratio

  subroutine evaluate_error_and_derivatives(x,netouts,error,deriv)
    real(k_rd),dimension(:),intent(in)::x
    real(k_rd),dimension(:),intent(in)::netouts
    real(k_rd),intent(out),dimension(size(x,1))::error
    real(k_rd),dimension(size(netouts,1),size(x,1)),intent(out)::deriv
    real(k_rd),dimension(size(netouts,1)/3)::wts,means,stds
    real(k_rd),dimension(size(netouts,1)/3,size(x,1))::ratio
    integer::ikern,nkernels
    nkernels=size(netouts,1)/3
    wts = netouts(1:3*nkernels-2:3)
    wts = exp(wts)/sum(exp(wts))
    means = netouts(2:3*nkernels-1:3)
    stds = netouts(3:3*nkernels:3)
    stds = exp(stds)
    error = -log(evaluate_gmm(x,wts,means,stds))
    call evaluate_gaussian_ratio(x,wts,means,stds,ratio)
    do ikern = 1,nkernels
      deriv( 3*(ikern-1) + 1,:) = - wts(ikern)*(ratio(ikern,:)-1.0_k_rd)
      deriv( 3*(ikern-1) + 2,:) = - wts(ikern)*(x-means(ikern))*ratio(ikern,:)/stds(ikern)**2.
      deriv( 3*(ikern-1) + 3,:) = - wts(ikern)*(((x-means(ikern))/stds(ikern))**2. -1.0_k_rd )*ratio(ikern,:)
    end do
  end subroutine evaluate_error_and_derivatives

  function evaluate_mdn(mlp,inputs,x) result(p)
    type(mlp_network),intent(in)::mlp
    real(k_rd),dimension(:,:),intent(in)::inputs !inputs(:,neg)
    real(k_rd),dimension(:),intent(in)::x !x(nx)
    real(k_rd),dimension(size(x,1),size(inputs,2))::p !p(nx,neg)
    real(k_rd),dimension(get_layer_nout(mlp,mlp%nlayers),size(inputs,2))::netouts
    integer::ieg

    call evaluate_mlp(mlp,inputs,netouts)
    do ieg = 1,size(inputs,2)
      p(:,ieg) =  evaluate_gmm_from_network_outputs(x,netouts(:,ieg))
    end do
  end function evaluate_mdn

  subroutine evaluate_mdn_error_and_derivatives(mlp,inputs,targets,E,dEdp)
    type(mlp_network),intent(in)::mlp
    real(k_rd),dimension(:,:),intent(in)::inputs !inputs(:,neg)
    real(k_rd),dimension(size(inputs,2)),intent(in)::targets !targets(neg)
    real(k_rd),intent(out)::E
    real(k_rd),intent(out),dimension(mlp%nparams),optional::dEdp !dEdp(mlp%nparams)
    real(k_rd),dimension(:,:),allocatable::mlp_internal,mlp_internal_deriv
    real(k_rd),dimension(get_layer_nout(mlp,mlp%nlayers),size(inputs,2))::outputs
    real(k_rd),dimension(get_layer_nout(mlp,mlp%nlayers),size(inputs,2)),target::dEk_dnetout
    real(k_rd),dimension(1,size(inputs,2))::Ek
    integer::i1,i2,ilayer,n_internal,ieg
    real(k_rd),pointer,dimension(:,:)::p2dEdx,p2tmp

    n_internal = 0
    do ilayer = 1,mlp%nlayers
      n_internal = n_internal + get_layer_nout(mlp,ilayer)
    end do
    allocate(mlp_internal(n_internal,size(inputs,2)),mlp_internal_deriv(n_internal,size(inputs,2)))

    call evaluate_mlp(mlp,inputs,outputs,mlp_internal,mlp_internal_deriv)
    do ieg = 1,size(inputs,2)
      call evaluate_error_and_derivatives(targets(ieg:ieg),outputs(:,ieg),Ek(:,ieg),dEk_dnetout(:,ieg:ieg))
    end do
    E = sum(Ek)
    if(present(dEdp)) then
      i1 = n_internal+1
      allocate(p2dEdx(get_layer_nout(mlp,mlp%nlayers),size(inputs,2)))
      p2dEdx = dEk_dnetout
      do ilayer = mlp%nlayers,2,-1
        i2 = i1-1
        i1 = i2 - get_layer_nout(mlp,ilayer)+1
        dEdp(mlp%ib(1,ilayer):mlp%ib(2,ilayer)) = sum(p2dEdx*mlp_internal_deriv(i1:i2,:),2)
        dEdp(mlp%iw(1,ilayer):mlp%iw(2,ilayer)) = reshape(matmul(p2dEdx*mlp_internal_deriv(i1:i2,:), &
                                                    transpose(mlp_internal(i1-get_layer_nout(mlp,ilayer-1):i1-1,:))), &
                                                    (/mlp%iw(2,ilayer)-mlp%iw(1,ilayer)+1/))
        allocate(p2tmp(get_layer_nout(mlp,ilayer-1),size(inputs,2)))
        p2tmp = matmul(transpose(reshape(mlp%params(mlp%iw(1,ilayer):mlp%iw(2,ilayer)), &
                      (/get_layer_nout(mlp,ilayer),get_layer_nin(mlp,ilayer)/))),p2dEdx*mlp_internal_deriv(i1:i2,:))
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
    end if
    deallocate(mlp_internal,mlp_internal_deriv)
  end subroutine evaluate_mdn_error_and_derivatives

  subroutine train_mdn(mlp,inputs,targets,noise_std,maxit,monitor_inputs,monitor_targets,best_params,best_E)
    type(mlp_network),intent(inout)::mlp
    real(k_rd),dimension(:,:),intent(in)::inputs
    real(k_rd),dimension(size(inputs,2)),intent(in)::targets
    real(k_rd),dimension(size(inputs,1),size(inputs,2))::noise
    real(k_rd),intent(in)::noise_std
    integer,intent(in)::maxit
    real(k_rd),intent(in),dimension(:,:),optional::monitor_inputs
    real(k_rd),intent(in),dimension(:),optional::monitor_targets
    real(k_rd),intent(inout),dimension(mlp%nparams),optional::best_params
    real(k_rd),intent(inout),optional::best_E
    integer,  parameter    :: m = 5, iprint = -1
    real(k_rd), parameter    :: factr = 0.0_k_rd,pgtol=0.0_k_rd!factr  = 1.0d+7, pgtol  = 1.0d-5
    character(len=60)      :: task, csave
    logical,dimension(4)   :: lsave
    integer,dimension(44)  :: isave
    real(k_rd)               :: E,Emon
    real(k_rd),dimension(29) :: dsave
    integer,dimension(mlp%nparams)::nbound
    real(k_rd),dimension(mlp%nparams)::grad
    integer,dimension(3*mlp%nparams)::iwa
    real(k_rd),dimension(2*m*mlp%nparams + 5*mlp%nparams + 11*m*m + 8*m)::wa
    real(k_rd), allocatable  :: lower(:), upper(:)
    integer::i,j
    nbound = 0


    call evaluate_mdn_error_and_derivatives(mlp,inputs,targets,E,grad)
    if(present(monitor_inputs)) then
      if(.not.((present(monitor_targets) .and. present(best_params) .and. present(best_E)))) stop 'Monitor error'
    end if

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
          call evaluate_mdn_error_and_derivatives(mlp,inputs+noise,targets,E,grad)
          if(present(monitor_inputs)) then
            call evaluate_mdn_error_and_derivatives(mlp,monitor_inputs,monitor_targets,Emon)
            if(Emon .lt. best_E) then
              !print *, isave(30)
              best_params = mlp%params
              best_E = Emon
            end if
          end if
        end if
      end if
    end do
  end subroutine train_mdn

  subroutine evaluate_initial_biases(targets,biases)
    real(k_rd),dimension(:),intent(in)::targets
    real(k_rd),dimension(:),intent(out)::biases
    integer::it_num,i,nkern
    integer,dimension(size(targets,1))::cluster
    real(k_rd),dimension(1,size(biases,1)/3)::cluster_center
    integer,dimension(size(biases,1)/3)::cluster_population
    real(k_rd),dimension(size(biases,1)/3)::cluster_energy
    nkern=size(biases,1)/3
    ! Initialise cluster centres uniformly throughout the prior range
    cluster_center(1,1)=minval(targets)
    cluster_center(1,nkern) = maxval(targets)
    do i=2,nkern-1
      cluster_center(1,i) = cluster_center(1,1)+real(i-1,k_rd)*(cluster_center(1,nkern)-cluster_center(1,1))/real(nkern-1,k_rd)
    end do
    ! Now run kmeans to update the clusters
    call kmeans_02 (1,size(targets,1),nkern,10000, it_num, reshape(targets,(/1,size(targets,1)/)), &
      cluster, cluster_center, cluster_population, cluster_energy )
    do i = 1,nkern
      biases(3*(i-1)+1) = log(real(cluster_population(i),k_rd))
      biases(3*(i-1)+2) = cluster_center(1,i)
      biases(3*(i-1)+3) = log(sqrt(cluster_energy(i)/real(cluster_population(i))))
    end do
  end subroutine evaluate_initial_biases

  subroutine store_mdn(file,nkernels,net)
    character(len=*),intent(in)::file
    integer,intent(in)::nkernels
    type(mlp_network),intent(in)::net
    integer::lu,i
    open(newunit=lu,file=file,form='unformatted',status='replace')
    write(lu) nkernels
    write(lu) net%nlayers
    do i=1,net%nlayers+1
      write(lu) net%node_counts(i)
    end do
    do i=1,net%nlayers
      write(lu) net%neuron_type(i)
    end do
    do i=1,net%nparams
      write(lu) net%params(i)
    end do
    close(lu)
  end subroutine store_mdn

  subroutine load_mdn(file,nkernels,net)
    character(len=*),intent(in)::file
    integer,intent(out)::nkernels
    type(mlp_network),intent(out)::net
    type(mlp_settings)::settings
    integer::lu,i
    open(newunit=lu,file=file,form='unformatted',status='old')
    read(lu) nkernels
    read(lu) settings%nhidden
    allocate(settings%node_counts(settings%nhidden),settings%neuron_type(settings%nhidden-1))
    settings%nhidden = settings%nhidden-2

    do i=1,settings%nhidden+2
      read(lu)net%node_counts(i)
    end do
    do i=1,settings%nhidden+1
      read(lu)net%neuron_type(i)
    end do
    call allocate_mlp(net,settings)
    do i=1,net%nparams
      read(lu) net%params(i)
    end do
    close(lu)
    deallocate(settings%node_counts,settings%neuron_type)
  end subroutine load_mdn

  subroutine create_mdn_committee(committee,nmembers,settings)
    integer,intent(in)::nmembers
    type(mdn_committee),intent(out)::committee
    type(mlp_settings),dimension(:),intent(in)::settings
    integer::i
    committee%nmembers=nmembers

    allocate(committee%member(nmembers),committee%weight(nmembers),committee%nkernels(nmembers))
    do i=1,nmembers
      call allocate_mlp(committee%member(i),settings(i))
      committee%nkernels(i) = committee%member(i)%node_counts(committee%member(i)%nlayers+1)/3
    end do
    committee%weight =0.
  end subroutine create_mdn_committee

  subroutine train_mdn_committee(committee,inputs,targets,noise_std,maxit,monitor_inputs,monitor_targets)
    type(mdn_committee),intent(inout)::committee
    real(k_rd),dimension(:,:),intent(in)::inputs
    real(k_rd),dimension(size(inputs,2)),intent(in)::targets
    real(k_rd),intent(in)::noise_std
    integer,intent(in)::maxit
    real(k_rd),dimension(:,:),intent(in)::monitor_inputs
    real(k_rd),dimension(size(monitor_inputs,2)),intent(in)::monitor_targets
    real(k_rd),dimension(:),allocatable::best_params
    real(k_rd)::best_E
    integer::ict
    print *, "Training",committee%nmembers,"committee members..."
!$OMP PARALLEL DO private(best_E,best_params)
    do ict = 1,committee%nmembers
      print *, ict
      best_E = 1.0e10
      allocate(best_params(committee%member(ict)%nparams))
      call train_mdn(committee%member(ict),inputs,targets,noise_std,maxit, &
                                                      monitor_inputs,monitor_targets,best_params,best_E)
      committee%member(ict)%params = best_params
      deallocate(best_params)
      committee%weight(ict) = exp(-best_E/real(size(monitor_inputs,2)))
    end do
!$OMP END PARALLEL DO
    committee%weight = committee%weight/sum(committee%weight)
    print *, "...done!"
  end subroutine train_mdn_committee


  function evaluate_mdn_committee(committee,inputs,x) result(p)
    type(mdn_committee),intent(in)::committee
    real(k_rd),dimension(:,:),intent(in)::inputs !inputs(:,neg)
    real(k_rd),dimension(:),intent(in)::x !x(nx)
    real(k_rd),dimension(size(x,1),size(inputs,2))::p !p(nx,neg)
    integer::ict

    p=0.
    do ict=1,committee%nmembers
      p = p + committee%weight(ict)*evaluate_mdn(committee%member(ict),inputs,x)
    end do
  end function evaluate_mdn_committee

  !From GCC website
  subroutine init_random_seed()

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
    print *, "Seed:",seed
    call random_seed(put=seed)
    call zigset(seed(1))
  end subroutine init_random_seed

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
end module mdn
