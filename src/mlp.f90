module mlp
  use ziggurat
  implicit none

  integer,parameter::k_rd = kind(1.d0)
  integer,parameter::NEURON_TYPE_LINEAR = 1
  integer,parameter::NEURON_TYPE_TANH = 2

  type mlp_settings
    integer :: nhidden! Layers of neurons
    integer,dimension(:),allocatable :: node_counts ! dimension nhidden+2
    integer,dimension(:),allocatable :: neuron_type ! dimension nhidden+1
  end type mlp_settings

  type mlp_network
    integer::nlayers
    integer::nparams ! Total number of free parameters
    real(k_rd),dimension(:),allocatable::params !All parameters governing network
    integer,dimension(:,:),allocatable::iw ! Indices of w matrix for a given layer
    integer,dimension(:,:),allocatable::ib !Indices of b for a given layer
    integer,dimension(:),allocatable::neuron_type ! dimension nlayers
    integer,dimension(:),allocatable::node_counts ! dimension nlayers+1
  end type mlp_network
contains
  pure integer function get_layer_nin(mlp,ilayer) result(nin)
    type(mlp_network),intent(in)::mlp
    integer,intent(in)::ilayer
    nin = mlp%node_counts(ilayer)
  end function get_layer_nin

  pure integer function get_layer_nout(mlp,ilayer) result(nout)
    type(mlp_network),intent(in)::mlp
    integer,intent(in)::ilayer
    nout = mlp%node_counts(ilayer+1)
  end function get_layer_nout


  subroutine allocate_mlp(mlp,settings)
    type(mlp_network),intent(inout)::mlp
    type(mlp_settings),intent(in)::settings
    integer:: i
    mlp%nlayers = settings%nhidden + 1
    mlp%nparams = 0
    allocate(mlp%iw(2,mlp%nlayers),mlp%ib(2,mlp%nlayers))
    do i=1,mlp%nlayers
      mlp%iw(1,i) = mlp%nparams+1
      mlp%iw(2,i) = mlp%nparams+(settings%node_counts(i)*settings%node_counts(i+1))
      mlp%ib(1,i) = mlp%nparams+(settings%node_counts(i)*settings%node_counts(i+1))+1
      mlp%ib(2,i) = mlp%nparams + ((settings%node_counts(i)+1)*settings%node_counts(i+1))
      mlp%nparams = mlp%nparams + ((settings%node_counts(i)+1)*settings%node_counts(i+1))
    end do
    allocate(mlp%params(mlp%nparams),mlp%neuron_type(mlp%nlayers),mlp%node_counts(mlp%nlayers+1))
    mlp%neuron_type = settings%neuron_type
    mlp%node_counts = settings%node_counts
  end subroutine allocate_mlp

  subroutine deallocate_mlp(mlp)
    type(mlp_network),intent(inout)::mlp
    integer::i
    mlp%nlayers = -1
    mlp%nparams = -1
    deallocate(mlp%params,mlp%iw,mlp%ib,mlp%neuron_type,mlp%node_counts)
  end subroutine deallocate_mlp

  subroutine initialise_mlp(mlp,final_bias)
    type(mlp_network),intent(inout)::mlp
    real(k_rd),dimension(mlp%node_counts(mlp%nlayers+1)),optional::final_bias
    integer::i,j,ilayer
    mlp%params = 0.0_k_rd
    do ilayer=1,mlp%nlayers
      do i = mlp%iw(1,ilayer), mlp%iw(2,ilayer)
        mlp%params(i) = rnor()/real(mlp%node_counts(ilayer)+1)
      end do
      do i = mlp%ib(1,ilayer),mlp%ib(2,ilayer)
        mlp%params(i)=rnor()/real(mlp%node_counts(ilayer)+1)
      end do
    end do
    if(present(final_bias)) mlp%params(mlp%ib(1,mlp%nlayers):mlp%ib(2,mlp%nlayers)) = final_bias
  end subroutine initialise_mlp

  subroutine evaluate_mlp(mlp,inputs,outputs,internal_values,internal_derivatives)
    type(mlp_network),intent(in)::mlp
    real(k_rd),dimension(:,:),intent(in)::inputs
    real(k_rd),dimension(get_layer_nout(mlp,mlp%nlayers),size(inputs,2)),intent(out)::outputs
    real(k_rd),dimension(:,:),intent(out),optional,target::internal_values,internal_derivatives
    real(k_rd),dimension(:,:),pointer::p2internal
    integer::n_internal,ilayer, negs, iin,nin,iout,nout

    negs=size(inputs,2)
    if(size(inputs,1) .ne. get_layer_nin(mlp,1)) stop 'dimension error'
    n_internal = 0
    do ilayer = 1,mlp%nlayers
      n_internal = n_internal + get_layer_nout(mlp,ilayer)
    end do

    if(present(internal_values)) then
      if( (size(internal_values,1) .lt. n_internal) .or. (size(internal_values,2) .lt. negs) ) stop 'dimension error'
      p2internal => internal_values
    else
      allocate(p2internal(n_internal,negs))
    end if

    if(present(internal_derivatives)) then
      if( (size(internal_derivatives,1) .lt. n_internal) .or. (size(internal_derivatives,2) .lt. negs) ) stop 'dimension error'
    end if

    iout = 1
    nin = get_layer_nin(mlp,1)
    nout = get_layer_nout(mlp,1)
    p2internal(iout:iout+nout-1,:) = matmul(reshape(mlp%params(mlp%iw(1,1):mlp%iw(2,1)),(/nout,nin/)),inputs) &
                                        + spread(mlp%params(mlp%ib(1,1):mlp%ib(2,1)),dim=2,ncopies=negs)
    if(present(internal_derivatives)) then
      select case(mlp%neuron_type(1))
      case(NEURON_TYPE_LINEAR)
        internal_derivatives(iout:iout+nout-1,:) = 1.0_k_rd
      case(NEURON_TYPE_TANH)
        internal_derivatives(iout:iout+nout-1,:) = 1.0 - tanh(p2internal(iout:iout+nout-1,:))**2.
      case default
        stop 'Unrecognised neuron type'
      end select
    end if
    select case(mlp%neuron_type(1))
    case(NEURON_TYPE_LINEAR)
      continue
    case(NEURON_TYPE_TANH)
      p2internal(iout:iout+nout-1,:) = tanh(p2internal(iout:iout+nout-1,:))
    case default
      stop 'Unrecognised neuron type'
    end select
    iin = 1
    nin = nout
    do ilayer=2,mlp%nlayers
      iout = iout + nout
      nout = get_layer_nout(mlp,ilayer)
      p2internal(iout:iout+nout-1,:) = matmul(reshape(mlp%params(mlp%iw(1,ilayer):mlp%iw(2,ilayer)),(/nout,nin/)), &
                                            p2internal(iin:iin+nin-1,:)) &
                                            + spread(mlp%params(mlp%ib(1,ilayer):mlp%ib(2,ilayer)),dim=2,ncopies=negs)
      if(present(internal_derivatives)) then
        select case(mlp%neuron_type(ilayer))
        case(NEURON_TYPE_LINEAR)
          internal_derivatives(iout:iout+nout-1,:) = 1.0_k_rd
        case(NEURON_TYPE_TANH)
          internal_derivatives(iout:iout+nout-1,:) = 1.0 - tanh(p2internal(iout:iout+nout-1,:))**2.
        case default
          stop 'Unrecognised neuron type'
        end select
      end if
      select case(mlp%neuron_type(ilayer))
      case(NEURON_TYPE_LINEAR)
        continue
      case(NEURON_TYPE_TANH)
        p2internal(iout:iout+nout-1,:) = tanh(p2internal(iout:iout+nout-1,:))
      case default
        stop 'Unrecognised neuron type'
      end select
      iin = iin + nin
      nin = nout
    end do
    outputs = p2internal(iout:iout+nout-1,:)
    if(.not.present(internal_values)) deallocate(p2internal)
  end subroutine evaluate_mlp


end module mlp
