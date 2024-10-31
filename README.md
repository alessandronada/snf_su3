# snf_su3
Stochastic Normalizing Flows for SU(3) pure gauge theory simulations in D=4

Developers:
Andrea Bulgarelli
Elia Cellini
Alessandro Nada

Main dependencies:
* pytorch
* numpy
* autograd (env needed: "pip install autograd")
* numdifftools (env needed: "pip install numdifftools")

## SHAPE
- configurations shape = (bs, D, T, L,..., L, N, N)
- initialization shape = (bs, D, T, L,..., L, N)
- random numbers shape = (bs, k(N), D, T, L,..., L)
With k(N=2)=6, k(N=3)=18
