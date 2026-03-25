import whisper
import json
import os
print("Loading model")
model = whisper.load_model("large-v2")
print("model loaded")
audios = os.listdir("videos")
for audio in audios: 
    if("_" in audio):
        print("Inside for loop")
        number = audio.split("_")[0]
        title = audio.split("_")[1]
        print(number, title)
        result = model.transcribe(audio = f"videos/{audio}",  
                              language="hi",
                              task="translate",
                              word_timestamps=False )
        chunks = []
        for segment in result["segments"]:
             chunks.append({"number": number, "title":title, "start": segment["start"], "end": segment["end"], "text": segment["text"]})
        chunks_with_metadata = {"chunks": chunks, "text": result["text"]}
        with open(f"jsons/{audio}.json", "w") as f:
            json.dump(chunks_with_metadata,f)
