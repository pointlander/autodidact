// Copyright 2025 The Autodidact Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bytes"
	"embed"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"math/rand"
	"strconv"
	"strings"

	"github.com/pointlander/gradient/tf64"
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = 1.0e-3
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

//go:embed iris.zip
var Iris embed.FS

// Fisher is the fisher iris data
type Fisher struct {
	Measures []float64
	Label    string
	Cluster  int
	Index    int
}

// Labels maps iris labels to ints
var Labels = map[string]int{
	"Iris-setosa":     0,
	"Iris-versicolor": 1,
	"Iris-virginica":  2,
}

// Inverse is the labels inverse map
var Inverse = [3]string{
	"Iris-setosa",
	"Iris-versicolor",
	"Iris-virginica",
}

// Load loads the iris data set
func Load() []Fisher {
	file, err := Iris.Open("iris.zip")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		panic(err)
	}

	fisher := make([]Fisher, 0, 8)
	reader, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		panic(err)
	}
	for _, f := range reader.File {
		if f.Name == "iris.data" {
			iris, err := f.Open()
			if err != nil {
				panic(err)
			}
			reader := csv.NewReader(iris)
			data, err := reader.ReadAll()
			if err != nil {
				panic(err)
			}
			for i, item := range data {
				record := Fisher{
					Measures: make([]float64, 4),
					Label:    item[4],
					Index:    i,
				}
				for ii := range item[:4] {
					f, err := strconv.ParseFloat(item[ii], 64)
					if err != nil {
						panic(err)
					}
					record.Measures[ii] = f
				}
				fisher = append(fisher, record)
			}
			iris.Close()
		}
	}
	return fisher
}

func main() {
	iris := Load()
	rng := rand.New(rand.NewSource(1))
	type Network struct {
		Others tf64.Set
		Set    tf64.Set
	}
	networks := make([]Network, 2)
	for n := range networks {
		networks[n].Others = tf64.NewSet()
		networks[n].Others.Add("input", 4, len(iris))
		networks[n].Others.Add("output", 3, len(iris))
		input := networks[n].Others.ByName["input"]
		output := networks[n].Others.ByName["output"]
		for _, row := range iris {
			input.X = append(input.X, row.Measures...)
			out := make([]float64, 3)
			out[Labels[row.Label]] = 1
			output.X = append(output.X, out...)
		}

		networks[n].Set = tf64.NewSet()
		networks[n].Set.Add("l1", 4, 8)
		networks[n].Set.Add("b1", 8)
		networks[n].Set.Add("l2", 16, 3)
		networks[n].Set.Add("b2", 3)

		for ii := range networks[n].Set.Weights {
			w := networks[n].Set.Weights[ii]
			if strings.HasPrefix(w.N, "b") {
				w.X = w.X[:cap(w.X)]
				w.States = make([][]float64, StateTotal)
				for ii := range w.States {
					w.States[ii] = make([]float64, len(w.X))
				}
				continue
			}
			factor := math.Sqrt(2.0 / float64(w.S[0]))
			for range cap(w.X) {
				w.X = append(w.X, rng.NormFloat64()*factor)
			}
			w.States = make([][]float64, StateTotal)
			for ii := range w.States {
				w.States[ii] = make([]float64, len(w.X))
			}
		}

		/*drop := .3
		dropout := map[string]interface{}{
			"rng":  rng,
			"drop": &drop,
		}*/

		l1 := tf64.Everett(tf64.Add(tf64.Mul(networks[n].Set.Get("l1"), networks[n].Others.Get("input")), networks[n].Set.Get("b1")))
		l2 := tf64.Add(tf64.Mul(networks[n].Set.Get("l2"), l1), networks[n].Set.Get("b2"))
		loss := tf64.Avg(tf64.Quadratic(networks[n].Others.Get("output"), l2))

		for iteration := range 1024 {
			pow := func(x float64) float64 {
				y := math.Pow(x, float64(iteration+1))
				if math.IsNaN(y) || math.IsInf(y, 0) {
					return 0
				}
				return y
			}

			networks[n].Others.Zero()
			networks[n].Set.Zero()
			l := tf64.Gradient(loss).X[0]
			if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
				fmt.Println(iteration, l)
				return
			}

			norm := 0.0
			for _, p := range networks[n].Set.Weights {
				for _, d := range p.D {
					norm += d * d
				}
			}
			norm = math.Sqrt(norm)
			b1, b2 := pow(B1), pow(B2)
			scaling := 1.0
			if norm > 1 {
				scaling = 1 / norm
			}
			for _, w := range networks[n].Set.Weights {
				for ii, d := range w.D {
					g := d * scaling
					m := B1*w.States[StateM][ii] + (1-B1)*g
					v := B2*w.States[StateV][ii] + (1-B2)*g*g
					w.States[StateM][ii] = m
					w.States[StateV][ii] = v
					mhat := m / (1 - b1)
					vhat := v / (1 - b2)
					if vhat < 0 {
						vhat = 0
					}
					w.X[ii] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
				}
			}
			fmt.Println(l)
		}
	}
}
