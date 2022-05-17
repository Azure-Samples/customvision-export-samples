using System.CommandLine;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;


namespace CustomVision
{
    class Program
    {
        static async Task Main(string[] args)
        {
            var modelFilepathArgument = new Argument<FileInfo>("model_filepath");
            var imageFilepathArgument = new Argument<FileInfo>("image_filepath");
            var command = new RootCommand
            {
                modelFilepathArgument,
                imageFilepathArgument
            };

            command.SetHandler((FileInfo modelFilepath, FileInfo imageFilepath) => {
                Predict(modelFilepath, imageFilepath);
            }, modelFilepathArgument, imageFilepathArgument);

            await command.InvokeAsync(args);
        }

        static void Predict(FileInfo modelFilepath, FileInfo imageFilepath)
        {
            var session = new InferenceSession(modelFilepath.ToString());
            bool isBgr = session.ModelMetadata.CustomMetadataMap["Image.BitmapPixelFormat"] == "Bgr8";
            bool isRange255 = session.ModelMetadata.CustomMetadataMap["Image.NominalPixelRange"] == "NominalRange_0_255";
            var inputName = session.InputMetadata.Keys.First();
            int inputSize = session.InputMetadata[inputName].Dimensions[2];

            Tensor<float> tensor = LoadInputTensor(imageFilepath, inputSize, isBgr, isRange255);
            
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor<float>(inputName, tensor)
            };

            var resultsCollection = session.Run(inputs);
            var outputs = resultsCollection.First().AsTensor<float>();

            for (var i = 0; i < outputs.Length; i++) {
               Console.WriteLine("Label: {0}, Probability: {1}", i, outputs[0, i]);
            }
        }

        // Load an image file and create a tensor.
        static Tensor<float> LoadInputTensor(FileInfo imageFilepath, int imageSize, bool isBgr, bool isRange255)
        {
            var input = new DenseTensor<float>(new[] {1, 3, imageSize, imageSize});
            using (var image = Image.Load<Rgb24>(imageFilepath.ToString()))
            {
                image.Mutate(x => x.Resize(imageSize, imageSize));

                for (int y = 0; y < image.Height; y++)
                {
                    Span<Rgb24> pixelSpan = image.GetPixelRowSpan(y);
                    for (int x = 0; x < image.Width; x++)
                    {
                        if (isBgr)
                        {
                            input[0, 0, y, x] = pixelSpan[x].B;
                            input[0, 1, y, x] = pixelSpan[x].G;
                            input[0, 2, y, x] = pixelSpan[x].R;
                        }
                        else
                        {
                            input[0, 0, y, x] = pixelSpan[x].R;
                            input[0, 1, y, x] = pixelSpan[x].G;
                            input[0, 2, y, x] = pixelSpan[x].B;
                        }

                        if (!isRange255)
                        {
                            input[0, 0, y, x] = input[0, 0, y, x] / 255;
                            input[0, 1, y, x] = input[0, 1, y, x] / 255;
                            input[0, 2, y, x] = input[0, 2, y, x] / 255;
                        }
                    }
                }
            }
            return input;
        }
    }
}
