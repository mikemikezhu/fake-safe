import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from abc import ABC, abstractmethod


class AbstractSampleDisplayer(ABC):

    @abstractmethod
    def display_samples(self, name, samples,
                        should_display_directly,
                        should_save_to_file,
                        labels=None):
        raise NotImplementedError('Abstract class shall not be implemented')


class SampleImageDisplayer(AbstractSampleDisplayer):

    def __init__(self, row=1, column=1, cmap=None):
        self.row = row
        self.column = column
        self.cmap = cmap

    def display_samples(self, name, samples,
                        should_display_directly,
                        should_save_to_file,
                        labels=None):

        # Display image samples
        fig, axs = plt.subplots(self.row, self.column)

        count = 0
        for i in range(self.row):
            for j in range(self.column):
                axs[i, j].imshow(samples[count], cmap=self.cmap)
                axs[i, j].axis('off')
                if labels is not None:
                    axs[i, j].text(0.5, -0.15, labels[count],
                                   size=6, ha='center',
                                   transform=axs[i, j].transAxes)
                count += 1

        if should_display_directly:
            plt.show()

        if should_save_to_file:
            fig.savefig('output/{}.png'.format(name))
            plt.close()


class SampleDiagramDisplayer(AbstractSampleDisplayer):

    def display_samples(self, name, samples,
                        should_display_directly,
                        should_save_to_file,
                        labels=None):

        # Display diagram samples
        plt.title(name)
        plt.plot(samples)

        if should_display_directly:
            plt.show()

        if should_save_to_file:
            plt.savefig('output/{}.png'.format(name))
            plt.close()


class SampleTextDisplayer(AbstractSampleDisplayer):

    def display_samples(self, name, samples,
                        should_display_directly,
                        should_save_to_file,
                        labels=None):

        # Display text samples
        if should_display_directly:
            print('\n'.join(samples))

        if should_save_to_file:
            output_path = 'output/{}.txt'.format(name)
            with open(output_path, 'w') as data_file:
                for sample in samples:
                    data_file.write(sample + '\n')


class SampleConfusionMatrixDisplayer(AbstractSampleDisplayer):

    def display_samples(self, name, samples,
                        should_display_directly,
                        should_save_to_file,
                        labels=None):

        # Display confusion matrix
        fig, ax = plot_confusion_matrix(conf_mat=samples)
        plt.title(name)

        if should_display_directly:
            plt.show()

        if should_save_to_file:
            plt.savefig('output/{}.png'.format(name))
            plt.close()


class SampleReportDisplayer(AbstractSampleDisplayer):

    def display_samples(self, name, samples,
                        should_display_directly,
                        should_save_to_file,
                        labels=None):

        # Display report
        report_text = ''
        for key, value in samples.items():
            value = value if isinstance(value, str) else str(value)
            report_text = report_text + key + ': \n' + value + '\n'

        if should_display_directly:
            print(report_text)

        if should_save_to_file:
            output_path = 'output/{}.txt'.format(name)
            with open(output_path, 'w') as data_file:
                data_file.write(report_text)
