import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class AbstractSampleDisplayer(ABC):

    @abstractmethod
    def display_samples(self, name, sample,
                        should_display_directly,
                        should_save_to_file):
        raise NotImplementedError('Abstract class shall not be implemented')


class SampleImageDisplayer(AbstractSampleDisplayer):

    def __init__(self, row=1, column=1, cmap=None):
        self.row = row
        self.column = column
        self.cmap = cmap

    def display_samples(self, name, samples,
                        should_display_directly,
                        should_save_to_file):

        # Display image samples
        fig, axs = plt.subplots(self.row, self.column)

        count = 0
        for i in range(self.row):
            for j in range(self.column):
                axs[i, j].imshow(samples[count], cmap=self.cmap)
                axs[i, j].axis('off')
                count += 1

        if should_display_directly:
            plt.show()

        if should_save_to_file:
            fig.savefig('output/{}.png'.format(name))
            plt.close()


class SampleTextDisplayer(AbstractSampleDisplayer):

    def display_samples(self, name, samples,
                        should_display_directly,
                        should_save_to_file):

        # Display text samples
        if should_display_directly:
            print('\n'.join(samples))

        if should_save_to_file:
            output_path = 'output/{}.txt'.format(name)
            with open(output_path, 'w') as data_file:
                for sample in samples:
                    data_file.write(sample + '\n')
