# Third-party dependencies and local overlays

외부 저장소 전체와 model weight는 이 저장소에 vendoring하지 않습니다. 아래 commit을 checkout한 뒤 `third_party_overlays/third_party/<name>/`의 파일을 동일한 상대 경로에 복사하면 당시 로컬 수정 상태를 복원할 수 있습니다. 각 프로젝트의 라이선스와 model access 조건이 우선합니다.

| 이름 | upstream | 고정 commit | 로컬 변경 |
|---|---|---|---|
| DiffSynth-Studio | `https://github.com/modelscope/DiffSynth-Studio.git` | `fb337fbb90945ff829de69dbd44ded618f73e889` | Wan action 주입과 pipeline |
| boundless-world-model | `https://github.com/boundless-large-model/boundless-world-model.git` | `44acfd1b06f35f365f02f7bb2fc5da6beafcd6bc` | 변경 없음; BWM 참조 구현 |
| cosmos-predict2.5 | `https://github.com/nvidia-cosmos/cosmos-predict2.5.git` | `a2c298b0a3df3778b973fe65e9e58877b292d8a7` | cluster CPU affinity 호환 처리 |
| cosmos-framework | `https://github.com/NVIDIA/cosmos-framework.git` | `5e02e643c458ce06c7232244271f567dce1dec7a` | SO100 action FD config/export/distributed 처리 |
| DreamZero | `https://github.com/dreamzero0/dreamzero.git` | `ab790c198fbce33503358efbbd4187ce9a89adf3` | native inference 호환 수정 |
| IRASim | `https://github.com/bytedance/IRASim.git` | `c72b6dade6fcd65971e0aa8ab49ea39b15108c90` | dtype/inference 호환 수정 |
| HMA | `https://github.com/liruiw/HMA.git` | `e3c89088fa82f2c208af6796485b4ec65ab318c5` | attention/tokenizer 호환 수정 |

예를 들어 DiffSynth-Studio를 복원하는 절차는 다음과 같습니다.

```bash
mkdir -p third_party
git clone https://github.com/modelscope/DiffSynth-Studio.git third_party/DiffSynth-Studio
git -C third_party/DiffSynth-Studio checkout fb337fbb90945ff829de69dbd44ded618f73e889
cp -a third_party_overlays/third_party/DiffSynth-Studio/. third_party/DiffSynth-Studio/
```

다른 저장소도 같은 방식으로 고정 commit에 overlay를 복사합니다. overlay는 전체 upstream의 재배포본이 아니라 이 프로젝트에서 실제 변경된 파일의 snapshot입니다. `cosmos-framework/uv.lock`의 로컬 환경 해상 결과는 재현성이 낮고 플랫폼 종속적이어서 포함하지 않았습니다.
