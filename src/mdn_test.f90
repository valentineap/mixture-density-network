program mdn_test
  use mdn
  implicit none

  integer,parameter :: negs = 1000
  integer,parameter :: nkernel=3


  integer,parameter :: model_dim = 3
  integer,parameter :: data_dim = 5
  real(k_rd),dimension(data_dim,model_dim)::A
  real(k_rd),dimension(3*nkernel)::biases
  real(k_rd),dimension(model_dim,negs)::models,mon_mods
  real(k_rd),dimension(data_dim,negs)::model_results,mon_res
  real(k_rd),dimension(negs)::targets,mon_tar
  real(k_rd)::Emon
  real(k_rd),dimension(:),allocatable::best_params
  type(mlp_settings)::settings
  type(mlp_network)::net
  integer::i,ieg
  integer,parameter::nx = 1000
  real(k_rd),dimension(nx)::x
  real(k_rd),dimension(nx,1)::p

  call init_random_seed()
  call random_number(A)
  call random_number(models)
  call random_number(mon_mods)
  models = (2.0_k_rd*models)-1.0_k_rd
  mon_mods = (2.0_k_rd*mon_mods) - 1.0_k_rd
  model_results = matmul(A,models)
  mon_res = matmul(A,mon_mods)
  targets = models(1,:)
  mon_tar = mon_mods(1,:)

  settings%nhidden = 1
  allocate(settings%node_counts(3),settings%neuron_type(2))
  settings%node_counts = (/data_dim,50,3*nkernel/)
  settings%neuron_type = (/NEURON_TYPE_TANH,NEURON_TYPE_LINEAR/)
  call evaluate_initial_biases(nkernel,negs,targets,biases)
  call allocate_mlp(net,settings)
  call initialise_mlp(net,biases)
  print *, "Bias: ",biases
  allocate(best_params(net%nparams))



  Emon = 0.
  call train_mdn(nkernel,net,negs,model_results,targets,0.01_k_rd,1000,mon_res,mon_tar,best_params,Emon)
  print *, Emon
  net%params = best_params
  do i=1,nx
    x(i)=real(i-1,k_rd)/real(nx-1,k_rd)
  end do
  x = 2.0_k_rd*x - 1.0_k_rd
  call random_number(models)
  models = (2.0_k_rd*models)-1.0_k_rd
  model_results = matmul(A,models)
  call evaluate_mdn(nkernel,net,1,model_results(:,1:1),nx,x,p)
  open(unit=7,file='results.dat')
  do i = 1,nx
    write(7,*) x(i),p(i,1)
  end do
  print *, models(:,1)
  print *, sum(p(:,1))*2.0_k_rd/real(nx-1)

end program mdn_test
