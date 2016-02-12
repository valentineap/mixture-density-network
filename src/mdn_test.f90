program mdn_test
  use mdn
  implicit none

  integer,parameter :: negs = 33400,nmonegs=130,nveregs=130
  integer,parameter::maxnkernel=8
  integer:: nkernel


  integer,parameter :: model_dim = 28
  integer,parameter :: data_dim = 1
  !real(k_rd),dimension(data_dim,model_dim)::A
  real(k_rd),dimension(3*maxnkernel)::biases
  real(k_rd),dimension(model_dim,negs)::models
  real(k_rd),dimension(model_dim,nmonegs)::mon_mods
  real(k_rd),dimension(model_dim,nveregs)::ver_mods
  !real(k_rd),dimension(data_dim,negs)::model_results,mon_res
  real(k_rd),dimension(negs)::targets
  real(k_rd),dimension(nmonegs)::mon_tar
  real(k_rd),dimension(nveregs)::ver_tar
  real(k_rd)::Emon,mean,std,rand
  real(k_rd),dimension(:),allocatable::best_params
  integer,parameter::nct = 250
  type(mlp_settings),dimension(nct)::settings
  !type(mlp_network)::net
  type(mdn_committee)::committee
  integer::i,ieg,j,lu,tpar,imon,nhid,nkern
  integer,parameter::nx = 1000
  real(k_rd),dimension(nx)::x
  real(k_rd),dimension(nx,1)::p
  real(k_rd),dimension(:,:),allocatable::tmp,tmp2,tmp3
  integer::ict


  call init_random_seed()
  allocate(tmp(negs,40),tmp2(nmonegs,40),tmp3(nveregs,40))
  open(newunit=lu,file='training.dat',status='old')
  do i =1,33400
    read(lu,*) (tmp(i,j),j=1,40)
  end do
  close(lu)
  open(newunit=lu,file='test.dat',status='old')
  do i=1,130
    read(lu,*) (tmp2(i,j),j=1,40)
  end do
  close(lu)
  open(newunit=lu,file='verification.dat',status='old')
  do i=1,130
    read(lu,*) (tmp3(i,j),j=1,40)
  end do
  close(lu)

  do i=1,model_dim
    mean = sum(tmp(:,i))/real(negs,k_rd)
    std = sqrt(sum((tmp(:,i)-mean)**2.)/real(negs-1,k_rd))
    models(i,:) = (tmp(:,i) - mean)/std
    mon_mods(i,:) = (tmp2(:,i)-mean)/std
    ver_mods(i,:) = (tmp3(:,i)-mean)/std
  end do
  tpar = 9
  mean = sum((tmp(:,28+tpar)))/real(negs,k_rd)
  std = sqrt(sum(((tmp(:,28+tpar))-mean)**2.)/real(negs-1,k_rd))
  targets = ((tmp(:,28+tpar))-mean)/std
  mon_tar = ((tmp2(:,28+tpar))-mean)/std
  ver_tar = ((tmp3(:,28+tpar))-mean)/std
  deallocate(tmp,tmp2,tmp3)
  !call random_number(A)
  !call random_number(models)
  !call random_number(mon_mods)
  !models = (2.0_k_rd*models)-1.0_k_rd
  !mon_mods = (2.0_k_rd*mon_mods) - 1.0_k_rd
  !model_results = matmul(A,models)
  !mon_res = matmul(A,mon_mods)
  !targets = models(1,:)
  !mon_tar = mon_mods(1,:)

  do ict=1,nct
    settings(ict)%nhidden = 1
    allocate(settings(ict)%node_counts(3),settings(ict)%neuron_type(2))
    call random_number(rand)
    nkernel = int(3+rand*5)
    call random_number(rand)
    nhid = int(20+rand*20)
    settings(ict)%node_counts = (/model_dim,nhid,3*nkernel/)
    settings(ict)%neuron_type = (/NEURON_TYPE_TANH,NEURON_TYPE_LINEAR/)
  end do
  call create_mdn_committee(committee,nct,settings)

  !call allocate_mlp(net,settings)
  do ict=1,nct
    call evaluate_initial_biases(targets,biases)
    call initialise_mlp(committee%member(ict),biases(1:3*committee%nkernels(ict)))
  end do
  !allocate(best_params(net%nparams))



  !Emon = 1e99_k_rd
  call train_mdn_committee(committee,models,targets,0.00_k_rd,50,mon_mods,mon_tar)
  ! print *, Emon
  !net%params = best_params
  do i=1,nx
   x(i)=real(i-1,k_rd)/real(nx-1,k_rd)
  end do
  x = 4.0_k_rd*x - 2.0_k_rd
  ! call random_number(models)
  ! models = (2.0_k_rd*models)-1.0_k_rd
  ! model_results = matmul(A,models)
  open(unit=7,file='results.dat')
  write(7,*) nx
  write(7,*) (mean+std*x(i),i=1,nx)
  do imon = 1,nveregs
    p = evaluate_mdn_committee(committee,ver_mods(:,imon:imon),x)
    write(7,*) mean+std*ver_tar(imon),(p(i,1),i=1,nx)
  end do



end program mdn_test
