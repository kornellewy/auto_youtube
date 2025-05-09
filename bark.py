from transformers import AutoProcessor, AutoModel
import scipy

processor = AutoProcessor.from_pretrained("suno/bark")
model = AutoModel.from_pretrained("suno/bark")

inputs = processor(
    text=[
        """
In 1950, computer scientist Alan Turing published his landmark paper "Computing Machinery and Intelligence," introducing what would later be known as the Turing Test.
This test proposed a way to determine if a machine could exhibit intelligent behavior indistinguishable from a human. 
Turing's revolutionary idea suggested that if a human evaluator couldn't reliably tell the difference between responses from a machine and a human, the machine could be considered "intelligent."
This concept became the philosophical foundation for artificial intelligence research and still influences how we evaluate AI systems today, including modern language models."""
    ],
    return_tensors="pt",
    voice_preset="v2/en_speaker_6",
)

speech_values = model.generate(**inputs, do_sample=True)


sampling_rate = model.config.codec_config.sampling_rate
scipy.io.wavfile.write(
    "./bark_out.wav", rate=sampling_rate, data=speech_values.cpu().numpy().squeeze()
)
