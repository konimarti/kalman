package kalman

// https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/

import (
	"fmt"

	"github.com/konimarti/lti"
	"gonum.org/v1/gonum/mat"
)

//Context contains the current state and covariance of the system
type Context struct {
	X *mat.VecDense // Current system state
	P *mat.Dense    // Current covariance matrix
}

//Noise represents the measurement and system noise
type Noise struct {
	Q *mat.Dense // (discretized) system noise
	R *mat.Dense // measurement noise
}

//NewZeroNoise initializes a Noise struct
//q: dimension of square matrix Q
//r: dimension of square matrix R
func NewZeroNoise(q, r int) Noise {
	nse := Noise{
		Q: mat.NewDense(q, q, nil),
		R: mat.NewDense(r, r, nil),
	}
	return nse
}

//Filter interface for using the Kalman filter
type Filter interface {
	Apply(ctx *Context, z, ctrl *mat.VecDense) mat.Vector
	State() mat.Vector
}

//filtImpl is the implementation of the filter interface
type filterImpl struct {
	Lti        lti.Discrete
	Nse        Noise
	savedState *mat.VecDense
}

//NewFilter returns a Kalman filter
func NewFilter(lti lti.Discrete, nse Noise) Filter {
	return &filterImpl{lti, nse, nil}
}

//Apply implements the Filter interface
func (f *filterImpl) Apply(ctx *Context, z, ctrl *mat.VecDense) mat.Vector {
	// correct state and covariance
	err := f.Update(ctx, z, ctrl)
	if err != nil {
		fmt.Println(err)
	}

	// get response of system y
	filtered := f.Lti.Response(ctx.X, ctrl)

	// save current context state
	f.savedState = mat.VecDenseCopyOf(ctx.X)

	// predict new state
	f.NextState(ctx, ctrl)

	// predict new covariance matrix
	f.NextCovariance(ctx)

	return filtered
}

//NextState
func (f *filterImpl) NextState(ctx *Context, ctrl *mat.VecDense) error {
	// X_k = Ad * X_k-1 + Bd * ctrl
	ctx.X = f.Lti.Predict(ctx.X, ctrl)
	return nil
}

//NextCovariance
func (f *filterImpl) NextCovariance(ctx *Context) error {
	// P_new = Ad * P * Ad^t + Q
	//ctx.P.Product(f.Lti.Ad, ctx.P, f.Lti.Ad.T())
	//ctx.P.Add(ctx.P, f.Nse.Q)
	ctx.P = lti.NewCovariance(f.Lti.Ad).Predict(ctx.P, f.Nse.Q)
	return nil
}

//Update performs Kalman update
func (f *filterImpl) Update(ctx *Context, z, ctrl mat.Vector) error {
	// kalman gain
	// K = P C^T (C P C^T + R)^-1
	var K, kt, PCt, CPCt, denom mat.Dense
	PCt.Mul(ctx.P, f.Lti.C.T())
	CPCt.Mul(f.Lti.C, &PCt)
	denom.Add(&CPCt, f.Nse.R)

	// calculation of Kalman gain with mat.Solve(..)
	// K = P C^T (C P C^T + R)^-1
	// K * (C P C^T + R) = P C^T
	// (C P C^T + R)^T K^T = (P C^T )^T
	err := kt.Solve(denom.T(), PCt.T())
	if err != nil {
		//log.Println(err)
		//log.Println("setting Kalman gain to zero")
		denom.Zero()
		K.Product(ctx.P, f.Lti.C.T(), &denom)
	} else {
		K.CloneFrom(kt.T())
	}

	// update state
	// X~_k = X_k + K * [z_k - C * X_k - D * ctrl ]
	var CXk, DCtrl, bracket, Kupd mat.VecDense
	CXk.MulVec(f.Lti.C, ctx.X)
	DCtrl.MulVec(f.Lti.D, ctrl)
	bracket.SubVec(z, &CXk)
	bracket.SubVec(&bracket, &DCtrl)
	Kupd.MulVec(&K, &bracket)
	ctx.X.AddVec(ctx.X, &Kupd)

	// update covariance
	// P~_k = P_k - K * [C * P_k]
	var KCP mat.Dense
	KCP.Product(&K, f.Lti.C, ctx.P)
	ctx.P.Sub(ctx.P, &KCP)

	return nil
}

//State return the current state of the context
func (f *filterImpl) State() mat.Vector {
	var state mat.VecDense
	state.CloneVec(f.savedState)
	return &state
}
