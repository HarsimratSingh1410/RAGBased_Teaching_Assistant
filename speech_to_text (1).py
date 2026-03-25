import whisper
import json
print("Process started\n")
model=whisper.load_model("large-v2")
print("model loaded\n")
result=model.transcribe(audio="videos/sample.mp3"
                        ,language="hi",
                        task="translate",verbose=True)
# with open("output.json","w") as f:
#     json.dump(f,result)
print("Transxribing completed\n")
chunks=[]
for segment in result['segments']:
    chunks.append({'id':segment['id'],'start':segment['start'],'end':segment['end'],'text':segment['text']})
print(chunks)
