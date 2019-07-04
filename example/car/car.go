package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"

	"github.com/konimarti/kalman"
	"github.com/konimarti/lti"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// prepare output file
	file, err := os.Create("car.csv")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	fmt.Fprintln(file, "Measured_v_x,Measured_v_y,Filtered_v_x,Filtered_v_y")

	ctx := kalman.Context{
		// init state: pos_x = 0, pox_y = 0, v_x = 30 km/h, v_y = 10 km/h
		X: mat.NewVecDense(4, []float64{0, 0, 30, 10}),
		// initial covariance matrix
		P: mat.NewDense(4, 4, []float64{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1}),
	}

	// time step
	dt := 0.1

	lti := lti.Discrete{
		// prediction matrix
		Ad: mat.NewDense(4, 4, []float64{
			1, 0, dt, 0,
			0, 1, 0, dt,
			0, 0, 1, 0,
			0, 0, 0, 1,
		}),
		// no external influence
		Bd: mat.NewDense(4, 4, nil),
		// scaling matrix for measurement
		C: mat.NewDense(2, 4, []float64{
			0, 0, 1, 0,
			0, 0, 0, 1,
		}),
		// scaling matrix for control
		D: mat.NewDense(2, 4, nil),
	}

	// G
	G := mat.NewDense(4, 2, []float64{
		0, 0,
		0, 0,
		1, 0,
		0, 1,
	})
	var Gd mat.Dense
	Gd.Mul(lti.Ad, G)

	// process model covariance matrix
	qk := mat.NewDense(2, 2, []float64{
		0.01, 0,
		0, 0.01,
	})
	var Q mat.Dense
	Q.Product(&Gd, qk, Gd.T())

	// measurement errors
	corr := 0.5
	R := mat.NewDense(2, 2, []float64{1, corr, corr, 1})

	// create noise struct
	nse := kalman.Noise{&Q, R}

	// create Kalman filter
	filter := kalman.NewFilter(lti, nse)

	// no control
	control := mat.NewVecDense(4, nil)

	for i := 0; i < 200; i++ {
		x1 := rand.NormFloat64()
		x2 := rand.NormFloat64()
		x3 := corr*x1 + math.Sqrt(1-corr)*x2
		y1 := 30.0 + 1.0*x1
		y2 := 10.0 + 1.0*x3
		// measure v_x and v_y with an error which is distributed according to stanard normal
		measurement := mat.NewVecDense(2, []float64{y1, y2})

		// apply filter
		filtered := filter.Apply(&ctx, measurement, control)

		// print out
		fmt.Fprintf(file, "%3.8f,%3.8f,%3.8f,%3.8f\n", measurement.AtVec(0), measurement.AtVec(1), filtered.AtVec(0), filtered.AtVec(1))
	}
}
