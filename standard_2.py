# #!/storage/LabJob/Projects/conda_env/s3prl_env/bin/python
# from fairseq_utils import dump_hubert_feature
import torch, torchaudio
import os, sys
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader
# dump_hubert_feature.main

torch.multiprocessing.set_sharing_strategy('file_system')

class MyResampler(torch.nn.Module):
    def __init__(self, expected_sample_rate=16000):
        super().__init__()
        self.expected_sample_rate = expected_sample_rate
        self.resamplers = {
            48000: torchaudio.transforms.Resample(  # for Common Voice
                48000, self.expected_sample_rate, 
                dtype=torch.float32),
            self.expected_sample_rate: lambda x: x,  # identity
        }
        
    def forward(self, waveform, waveform_rate):
        if waveform_rate == self.expected_sample_rate:
            return waveform, waveform.size(-1), self.expected_sample_rate
        resampled_waveform = self.resamplers[waveform_rate](waveform)
        return resampled_waveform, resampled_waveform.size(-1), self.expected_sample_rate
    
if not os.path.isdir('buffer'):
    os.mkdir('buffer')
    
class AudioDataset(Dataset):
    def __init__(self, datatable, root):
        self.datatable = datatable
        self.root = root
        self.resampler = MyResampler()
        self.other_cols = sorted(
            [col for col in self.datatable.columns if col not in ['audio', 'sr', 'n_frames']]
        )
        
    def __len__(self):
        return len(self.datatable)
    
    def __getitem__(self, idx):
        row = self.datatable.iloc[idx]
        name = Path(row['audio']).stem
        waveform, waveform_rate = torchaudio.load(str(self.root / (row['audio'])))
        assert waveform_rate == row['sr']
        # from IPython import embed; embed()
        if row['n_frames'] != waveform.size(-1) and False: 
            print("""N Frames should be {}, but get {} instead. ...""".format(row["n_frames"], waveform.size()))
        
        resampled_waveform, resampled_nframes, resampled_sr = self.resampler(waveform, waveform_rate)
        return name, resampled_waveform, resampled_nframes, resampled_sr, *[row[col] for col in self.other_cols]

# mydst = AudioDataset(
#     datatable=pd.read_csv('standard_0.tsv', sep='\t'), 
#     root=Path('/storage/LabJob/Projects/Data/CovoST4/cv-corpus-6.1-2020-12-11/en/clips'))

mydst = AudioDataset(
    datatable=pd.read_csv('/home/jeffeuxmartin/Projects/Mys3prl/s3prl/s3prl/data/covost_en_de/train.tsv', sep='\t'), 
    root=Path('/livingrooms/public/CoVoST2/cv-corpus-6.1-2020-12-11/en/clips'),
)

mydataloader = DataLoader(mydst, batch_size=1, shuffle=False, num_workers=eval(sys.argv[1]))
from tqdm import tqdm

# 將所有音訊儲存為 WAV 檔案

def dumpout(table, dumped):
    for i, sample in enumerate(tqdm(table, desc='dumping')):
        (name, audio, nsamples, sr, *others) = sample
        ([name], [audio], [nsamples], [sr]) = (name, audio, nsamples, sr)
        [others] = zip(*others)
        nsamples, sr = nsamples.item(), sr.item()
        newname = f'buffer/{name}.wav'
        torchaudio.save(newname, audio, sr)
        dumped.append([newname, nsamples, sr, *others])
    return dumped, []


audio_list = []  # 儲存所有音訊資料的列表
dumped = []  # 儲存已經處理過的音訊資料的列表
for batch in tqdm(mydataloader, desc='resampling'):
    audio_list.append(batch)  # 將批處理的資料新增到列表中
    # 這裡是你對音訊資料進行處理的程式碼
    if len(audio_list) >= 5000:
        dumped, audio_list = dumpout(audio_list, dumped)
dumped, audio_list = dumpout(audio_list, dumped)
assert audio_list == []
newtable = dumped


newtable = pd.DataFrame(newtable, columns=['audio', 'n_frames', 'sr', *mydst.other_cols])
newtable.to_csv('standardout_1.tsv', sep='\t', index=False)
