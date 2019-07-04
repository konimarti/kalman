package kalman

import (
	"fmt"
	"testing"

	"github.com/konimarti/lti"
	"gonum.org/v1/gonum/mat"
)

// Testing based on example on page 145 in book "Kalman Filter" by R. Marchthaler, 2017

//newContext
func newContext() *Context {
	// define current context
	ctx := Context{
		X: mat.NewVecDense(4, []float64{976.32452, 0, 0.092222, 0}),
		P: mat.NewDense(4, 4, []float64{
			3, 0, 0, 0,
			0, 3, 0, 0,
			0, 0, 3, 0,
			0, 0, 0, 0.03,
		}),
	}
	return &ctx
}

//newSetup is a helper functions for tests
func newSetup() (lti.Discrete, Noise) {

	// define LTI system
	dt := 0.1
	lti := lti.Discrete{
		Ad: mat.NewDense(4, 4, []float64{
			1, dt, 0.5 * dt * dt, 0,
			0, 1, dt, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		}),
		Bd: mat.NewDense(4, 1, nil),
		C: mat.NewDense(2, 4, []float64{
			1, 0, 0, 0,
			0, 0, 1, -1,
		}),
		D: mat.NewDense(2, 1, nil),
	}

	// define system and measurement noise
	q1 := 100.0 / 9.0
	q2 := 0.04 / 1000.0
	nse := Noise{
		Q: mat.NewDense(4, 4, []float64{
			0.25 * q1 * dt * dt * dt * dt, 0.5 * q1 * dt * dt * dt, 0.5 * q1 * dt * dt, 0,
			0.5 * q1 * dt * dt * dt, q1 * dt * dt, q1 * dt, 0,
			0.5 * q1 * dt * dt, q1 * dt, q1, 0,
			0, 0, 0, q2,
		}),
		R: mat.NewDense(2, 2, []float64{20, 0, 0, 0.2}),
	}

	return lti, nse
}

//NewImplementedFilter returns the implementation of the Kalman filter for testing
func newImplementedFilter() *filterImpl {
	lti, nse := newSetup()
	return &filterImpl{lti, nse, nil}
}

func TestPredictionState(t *testing.T) {
	ctx := newContext()
	filter := newImplementedFilter()

	// predict next state
	ctrl := mat.NewVecDense(1, nil)
	filter.NextState(ctx, ctrl)

	expectedVec := mat.NewVecDense(4, []float64{
		976.32498, 0.0092222, 0.092222, 0,
	})
	if !mat.EqualApprox(expectedVec, ctx.X, 1e-4) {
		fmt.Println("actual:", ctx.X)
		fmt.Println("expected:", expectedVec)
		t.Error("PredictState")
	}
}

func TestPredictionCovariance(t *testing.T) {
	ctx := newContext()
	filter := newImplementedFilter()

	// predict next covariance
	filter.NextCovariance(ctx)

	// predict next covariance
	expected := mat.NewDense(4, 4, []float64{
		3.0304, 0.30706, 0.070556, 0,
		0.30706, 3.1411, 1.4111, 0,
		0.070556, 1.4111, 14.111, 0,
		0, 0, 0, 0.03004,
	})
	if !mat.EqualApprox(expected, ctx.P, 1e-4) {
		fmt.Println("actual:", ctx.P)
		fmt.Println("expected:", expected)
		t.Error("PredictCovariance")
	}
}

func TestUpdate(t *testing.T) {
	ctx := newContext()
	filter := newImplementedFilter()

	ctrl := mat.NewVecDense(1, nil)
	z := mat.NewVecDense(2, []float64{
		976.32452, 0.092222,
	})
	if err := filter.Update(ctx, z, ctrl); err != nil {
		t.Error(err)
	}
	expectedX := mat.NewVecDense(4, []float64{
		976.32452, 0, 0.092222, 0,
	})
	if !mat.EqualApprox(expectedX, ctx.X, 1e-4) {
		fmt.Println("actual:", ctx.X)
		fmt.Println("expected:", expectedX)
		t.Error("UpdateState")
	}
}

func TestFilter(t *testing.T) {
	lti, nse := newSetup()
	ctx := newContext()
	filter := NewFilter(lti, nse)

	ctrl := mat.NewVecDense(1, nil)

	config := []struct {
		Iter     int
		Input    []float64
		Expected []float64
	}{
		{
			Iter: 1,
			Input: []float64{
				976.32, 0.092222,
			},
			Expected: []float64{
				976.32452, 0.092222202,
			},
		},
		{
			Iter: 2,
			Input: []float64{
				979.37006, 0.52210785,
			},
			Expected: []float64{
				976.6817722228133, 0.5147628306401388,
			},
		},
		{
			Iter: 3,
			Input: []float64{
				977.8754, 0.98211677,
			},
			Expected: []float64{
				976.8229728968552, 0.9740485904798598,
			},
		},
	}

	for _, cfg := range config {
		z := mat.NewVecDense(2, cfg.Input)
		filteredResult := filter.Apply(ctx, z, ctrl)
		expectedResult := mat.NewVecDense(2, cfg.Expected)
		if !mat.EqualApprox(expectedResult, filteredResult, 1e-4) {
			fmt.Println("actual:", filteredResult)
			fmt.Println("expected:", expectedResult)
			t.Error("ApplyFilter:", cfg.Iter)
		}
	}

}
