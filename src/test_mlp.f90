program test_mlp
  use mlp
  use ziggurat
  implicit none

  type(mlp_settings)::settings
  type(mlp_network)::network
  real(k_rd),dimension(10,20)::inputs
  real(k_rd),dimension(:,:),allocatable::outputs
  integer::i
  settings%nhidden = 1
  allocate(settings%node_counts(3),settings%neuron_type(2))
  settings%node_counts = (/10, 20, 3/)
  settings%neuron_type = (/NEURON_TYPE_TANH,NEURON_TYPE_LINEAR/)
  call random_number(inputs)
  call allocate_mlp(network,settings)
  call initialise_mlp(network)
  call evaluate_mlp(network,inputs,outputs)
  print *, outputs
  call deallocate_mlp(network)

end program test_mlp
