from laboratory.dloader.mnist_sample_loader import MnistSampleLoader
from laboratory.nn_models.conv_sample import ConvSample
from laboratory.trainer.sample_trainer import SampleTrainer

def main():
    d = MnistSampleLoader()
    model = ConvSample()
    trainer = SampleTrainer(model, d.load_test)
    trainer.training()

if __name__ == "__main__":
    main()