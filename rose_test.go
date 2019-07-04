package kalman

import (
	"fmt"
	"testing"

	"github.com/konimarti/lti"
	"gonum.org/v1/gonum/mat"
)

// Testing based on example on page 145 in book "Kalman Filter" by R. Marchthaler, 2017

//newContext for Rose Filter
func newRoseContext() *Context {
	// define current context
	ctx := Context{
		X: mat.NewVecDense(1, []float64{0.04280385872149909}),
		P: mat.NewDense(1, 1, []float64{0}),
	}
	return &ctx
}

//newRoseFilter is a helper functions for tests
func newRoseFilter() *roseImpl {

	// define LTI system
	rose := NewRoseFilter(
		lti.Discrete{
			Ad: mat.NewDense(1, 1, []float64{1}),
			Bd: mat.NewDense(1, 1, []float64{0}),
			C:  mat.NewDense(1, 1, []float64{1}),
			D:  mat.NewDense(1, 1, []float64{0}),
		},
		mat.NewDense(1, 1, []float64{1}), // Gd
		9.0,                              // Gamma
		0.5,                              // AlphaR
		0.3,                              // AlphaM
	)

	return rose.(*roseImpl)
}

func TestRoseFilter(t *testing.T) {
	ctx := newRoseContext()
	filter := newRoseFilter()

	ctrl := mat.NewVecDense(1, nil)

	// init filter for comparison testing
	y := filter.Std.Lti.Response(ctx.X, ctrl)
	filter.Rose.E1 = y
	filter.Rose.EE1.Mul(y, y.T())

	config := []struct {
		Iter     int
		Input    []float64
		Expected []float64
	}{
		{
			Iter: 1,
			Input: []float64{
				0.04280385872149909,
			},
			Expected: []float64{
				0.04280385872149909,
			},
		},
		{
			Iter: 2,
			Input: []float64{
				-0.09725182469943415,
			},
			Expected: []float64{
				0.04280385872149909,
			},
		},
		{
			Iter: 3,
			Input: []float64{
				0.002742478388650294,
			},
			Expected: []float64{
				0.04280385872149909,
			},
		},
	}

	for _, cfg := range config {
		z := mat.NewVecDense(1, cfg.Input)
		filteredResult := filter.Apply(ctx, z, ctrl)
		expectedResult := mat.NewVecDense(1, cfg.Expected)
		if !mat.EqualApprox(expectedResult, filteredResult, 1e-4) {
			fmt.Println("actual:", filteredResult)
			fmt.Println("expected:", expectedResult)
			t.Error("ApplyFilter:", cfg.Iter)
		}
	}

}
