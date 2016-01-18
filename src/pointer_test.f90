program pointer_test
  implicit none
  integer,dimension(:),allocatable::arr
  integer,pointer,dimension(:,:)::p1,p2
  integer::i

  allocate(p1(2,3))
  p1=0
  p1(2,2) = 3
  print *, p1
  p2 => p1
  print *, p2
  allocate(p1(4,4))
  p1(3,3)=8
  print *, p2
  print *, p1
  allocate(arr(10))
  do i=1,10
    arr(i)=i
  end do
  print *, arr,transpose(arr)
end program
