package main

import (
	"fmt"
	"image"
	_ "image/png"
	"io/ioutil"
	"log"
	"os"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {

	// Load a frozen graph to use for queries
	modelpath := "pupil_tf.pb"
	model, err := ioutil.ReadFile(modelpath)
	if err != nil {
		log.Fatal(err)
	}

	// Construct an in-memory graph from the serialized form.
	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		//if err := graph.ImportWithOptions(model, GraphImportOptions{"", "CPU"}); err != nil {
		log.Fatal(err)
	}

	// Create a session for inference over graph.
	session, err := tf.NewSession(graph, nil)

	if err != nil {
		log.Fatal(err)
	}

	defer session.Close()
	tensor2, _ := makeTensorFromImage("pictu.png")
	//fmt.Println(tensor2)
	//fmt.Println(tensor2.Value())
	fmt.Println("we are good")

	final, err := session.Run(
		map[tf.Output]*tf.Tensor{
			graph.Operation("input").Output(0): tensor2,
		},
		[]tf.Output{
			graph.Operation("output").Output(0),
		},
		nil)
	if err != nil {
		log.Fatal(err)
		fmt.Println("failed here")
	}
	fmt.Printf("Result value: %v \n", final[0].Value().([][]float32)[0])

}

func makeTensorFromImage(filename string) (*tf.Tensor, error) {
	const (
		// - The model was trained after with images scaled to 224x224 pixels.
		// - The colors, represented as R, G, B in 1-byte each were converted to
		//   float using (value - Mean)/Std.
		// If using a different pre-trained model, the values will have to be adjusted.
		H, W = 224, 224
		Mean = 0
		Std  = float32(255)
	)
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	img, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}
	// 4-dimensional input:
	// - 1st dimension: Batch size (the model takes a batch of images as
	//                  input, here the "batch size" is 1)
	// - 2nd dimension: Rows of the image
	// - 3rd dimension: Columns of the row
	// - 4th dimension: Colors of the pixel as (R, G, B)
	// Thus, the shape is [1, 3, 224, 224]
	var ret [1][3][H][W]float32
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			px := x + img.Bounds().Min.X
			py := y + img.Bounds().Min.Y
			r, g, b, _ := img.At(px, py).RGBA()
			ret[0][0][y][x] = float32((int(r>>8) - Mean)) / Std //float32((int(r>>8) - int(r>>8)) + 2) //
			ret[0][1][y][x] = float32((int(g>>8) - Mean)) / Std //float32(int(g>>8) - int(g>>8) + 1)   //
			ret[0][2][y][x] = float32((int(b>>8) - Mean)) / Std //float32(int(b>>8) - int(b>>8) + 1)   //

		}
	}
	return tf.NewTensor(ret)
}
