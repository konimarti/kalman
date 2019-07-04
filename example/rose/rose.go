package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"

	"github.com/konimarti/kalman"
	"github.com/konimarti/lti"
	"gonum.org/v1/gonum/mat"
)

func main() {

	// load data
	y := load("p133.csv")[0]

	// prepare output file
	file, err := os.Create("p133_out.csv")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	fmt.Fprintln(file, "m,f")

	// define LTI system
	lti := lti.Discrete{
		Ad: mat.NewDense(1, 1, []float64{1}),
		Bd: mat.NewDense(1, 1, nil),
		C:  mat.NewDense(1, 1, []float64{1}),
		D:  mat.NewDense(1, 1, nil),
	}

	// system noise / process model covariance matrix ("Systemrauschen")
	Gd := mat.NewDense(1, 1, []float64{1})

	ctx := kalman.Context{
		// initial state
		X: mat.NewVecDense(1, []float64{y[0]}),
		// initial covariance matrix
		P: mat.NewDense(1, 1, []float64{0}),
	}

	// create ROSE filter
	gammaR := 9.0
	alphaR := 0.5
	alphaM := 0.3
	filter := kalman.NewRoseFilter(lti, Gd, gammaR, alphaR, alphaM)

	// no control
	u := mat.NewVecDense(1, nil)

	for _, row := range y {
		// new measurement
		y := mat.NewVecDense(1, []float64{row})

		// apply filter
		filter.Apply(&ctx, y, u)

		// get corrected state vector
		state := filter.State()

		// print out input and output signals
		fmt.Fprintf(file, "%3.8f,%3.8f\n", y.AtVec(0), state.AtVec(0))
	}
}

func load(file string) [][]float64 {
	f, err := os.Open(file)
	if err != nil {
		panic("could not open file")
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.Comma = ','

	lines, err := r.ReadAll()
	if err != nil {
		panic("could read data from file")
	}

	y := make([][]float64, len(lines))
	for row, line := range lines {
		tmp := make([]float64, len(line))
		for i, entry := range line {
			var value float64
			if value, err = strconv.ParseFloat(entry, 64); err != nil {
				fmt.Println("error parsing", line, i, entry)
				panic("could not parse all values")
			}
			tmp[i] = value
		}
		y[row] = tmp
	}
	return y
}
