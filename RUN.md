# 常駐型AIコンパニオン — speech-to-speech 起動手順

`companion/README.md` の構成を speech-to-speech で動かす。
LLM は llama.cpp に外出しし、STT と TTS だけをパイプライン内に置く
（16GB機では MLX の LLM/STT/TTS 三点同居が GPU の作業セットを超えるため）。

## 1. LLM サーバー

```bash
./scripts/serve_llm.sh
```

"server is listening" を待つ。パイプラインは起動時に warmup で1回叩くので、
先に上がっていないと接続エラーで落ちる。

## 2. パイプライン

```bash
# 人格。companion/README.md の決定事項に対応:
#   「物語のない、声で話せる同居人」    -> 同居している親しい相手
#   「攻略も、クリアも、エンディングも無い」-> アシスタントとして振る舞わせない
#   まず受け止める / 助言を並べない       -> phase0 SYSTEM_PROMPT から踏襲
#
# 日本語は JapaneseClauseTokenizer が「。」「、」で切るようになったので、返答の長さは
# TTFA に乗らなくなった(以前は NLTK が「。」を文末と認識せず、返答全体が常に1文として
# 扱われ、LLM が生成し終わるまで TTS が始まらなかった)。40文字制約はそのための措置で、
# もう要らない。冒頭の短い反応句は依然として効く: 最初のクローズだけが TTFA の
# クリティカルパスに乗っている。
#
# phase0 の感情タグ [平][喜][哀][怒][驚][優] は入れない。speech-to-speech の
# remove_unspeechable / remove_markdown はどちらも角括弧を除去しないので、
# TTS がタグを読み上げてしまう。MetaHuman の受け手ができた段階で再検討する。
#
# 名前・年齢・職業・部屋の設定は未決（README.md の未決事項）。決まったらここに足す。
INIT_PROMPT='あなたはユーザーと同居している親しい相手です。恋人でも友人でもある距離感で、気取らない日本語の話し言葉で返します。アシスタントではありません。用件を尋ねたり、手伝いを申し出たりはしません。

話し方:
- 短い反応の一言から始める（「うん、」「そっか、」「へえ、」「えっ、」「まあ、」など）
- そのあと1〜2文
- まず受け止める。助言や説明を並べない

禁止: 英語、箇条書き、記号、絵文字、顔文字、マークダウン、長い前置き'

# 既定は 3。3クローズ溜まるまで TTS に渡らないので、日本語分割を入れても冒頭句が
# 即座に出ない。かつ _flush は " ".join(batch) でバッチを繋ぐので、日本語のクローズが
# 半角スペースで連結される。レイテンシと出力品質の両方の理由で --stream_batch_sentences
# を 1 にする。
# --text-input で、話しかける代わりに打ち込める。マイクは開いたまま。
# ASR は TTFA の最大要因(1,040ms / p50 2,154ms)で、幻聴もまだ残っている(下記「未解決」)。
# 打った場合はその両方を丸ごと迂回する。返事は音声のまま。
# 入力行はターミナル最下部に固定され、ログや文字起こしはその上を流れる。
# プロンプトが raw mode を取るので、Ctrl-C / Ctrl-D は入力行で押せば終了する。
./.venv/bin/speech-to-speech local \
  --text-input \
  --stt mlx-audio-whisper \
  --mlx_audio_whisper_model_name mlx-community/whisper-large-v3-turbo \
  --language ja \
  --llm_backend chat-completions \
  --responses_api_base_url http://127.0.0.1:8080/v1 \
  --model_name "unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M" \
  --init_chat_prompt "$INIT_PROMPT" \
  --stream_batch_sentences 1 \
  --tts qwen3 \
  --qwen3_tts_device mps \
  --qwen3_tts_model_name Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --qwen3_tts_mlx_quantization 6bit \
  --qwen3_tts_instruct "落ち着いた低めの女性の声で、ゆっくり親しげに話す。" \
  --qwen3_tts_gen_temperature 0 \
  --qwen3_tts_gen_repetition_penalty 1.5 \
  --qwen3_tts_language ja
```

### TTS のモデル種別

Qwen3-TTS は3種あり、声の決め方=デコードの条件付けが違う。ハンドラは
`process` で「クローン参照があるか」→ model_type の順に分岐するので、
種別を変えるときは前の種別の引数を必ず外す(残っていると別経路に落ちる)。

| モデル | 声の指定 | 必要な引数 |
|---|---|---|
| Base | 参照音声のクローン | `--qwen3_tts_ref_audio` + `--qwen3_tts_ref_text` |
| CustomVoice | プリセット話者 | `--qwen3_tts_speaker`(日本語は `ono_anna` のみ) |
| VoiceDesign | instruct のテキスト指示 | `--qwen3_tts_instruct` |

`Qwen/...` の ID は Apple Silicon で `mlx-community/...` に自動マップされる。
どの種別も `4bit` `5bit` `6bit` `8bit` `bf16` が mlx-community にある。

### 短い相槌が壊れる問題（解決済み）

CustomVoice と VoiceDesign は、2〜4文字の発話で EOS を出さずトークン上限まで
回り切っていた。「うん、」が28.8秒になり(CustomVoice)、あるいは12秒ぶんの
無音になった(VoiceDesign)。

原因は2つの積で、どちらも修正済み。

**1. トークン予算の下限。** `_estimate_max_new_tokens` は「うん、」に対して
正しく約1.9秒と見積もっていたのに、`MIN_QWEN3_TTS_UTTERANCE_TOKENS = 360`
(= 28.8秒)の下限がそれを上書きしていた。下限は「推定が低すぎて実発話が
切れる」のを防ぐためのものだが、EOS を出さないモデルではそれ自体が暴走の
予算になる。下限を「そのテキストが必要としうる長さ」(`QWEN3_RUNAWAY_ESTIMATE_MULTIPLE`
= 生の推定の2.5倍)で頭打ちにした。360 が妥当な長さの発話では下限はこれまで通り効く。

予算を絞ると逆に「実発話が切り詰められる」危険が出る。数字や大文字は1文字ずつ
読まれるため、推定の 14文字/秒 では大幅に足りない(実測 約3.4文字/秒)。
`ESTIMATED_QWEN3_SPELLED_CHARS_PER_SECOND` を足して推定式に反映し、さらに
`MIN_QWEN3_TTS_RUNAWAY_TOKENS = 80`(6.4秒)を最低限の余裕として置いた。

**2. greedy デコードの反復ループ。** `--qwen3_tts_gen_temperature 0` は
「声質が毎回変わる」を潰すために入れているが、プリセット話者や instruct には
音響プレフィックス(Base の参照音声にあたるもの)が無く、デコードを固定する
足場が無い。そのため短文で反復ループに落ちる。`--qwen3_tts_gen_repetition_penalty`
を追加した。1.3 でも大半は直るが「なるほど」が9.2秒残ったため 1.5 を採る。

修正後の実測(VoiceDesign 6bit, temperature 0, rp 1.5):

| | 修正前 | 修正後 | 予算の使用率 |
|---|---|---|---|
| 相槌10種の最悪 | 28.8s / 無音 | **6.18s** | 12〜96% |
| 通常の文3種の最悪 | 28.8s | **5.22s** | 35〜36% |
| 読み上げが長い入力4種の最悪 | — | **7.62s** | 43〜95% |

全ケースが自力で EOS を出して終端(予算1536で再生成しても出力が同一)。
無音は0件、全て決定的。**temperature 0 に戻せたので「声質が毎回変わる」問題も
再発しない。** CustomVoice 側も同時に直っている。

残る注意: 「あー、」6.18秒と「ABCDEFG」7.62秒は予算の95%前後を使う。
切り詰めは起きていないが余裕は薄い。

`repetition_penalty` は mlx バックエンド専用。faster-qwen3-tts (CUDA/CPU) では
無視され、設定すると警告が出る(temperature/top_k と同じ扱い)。Base の
voice-clone 経路は mlx-audio 側が `max(repetition_penalty, 1.5)` に強制するので、
1.5 は Base の既定と一致し、Base の出力は変わらない。

### 短い相槌を単独で TTS に流す設計は変えていない

`japanese_segmenter.py:41` の `first_min_chars=2` は、冒頭の2〜4文字を即座に
TTS へ流して TTFA を166ms稼ぐための意図的な設計。今回の修正はここに一切
触れていない。暴走時も発話そのものは正しくストリーミングされていて(「うん、」
28.77秒のうち実際の発話は先頭1.06秒)、壊れていたのは「言い終わったあとに
止まらない」ことだけだったため、止め方を直すだけで済んだ。

### モデルを Base に戻す場合

参照音声のクローン声質が必要なら:

```bash
  --qwen3_tts_model_name mlx-community/Qwen3-TTS-12Hz-1.7B-Base \
  --qwen3_tts_mlx_quantization 8bit \
  --qwen3_tts_gen_temperature 0 \
  --qwen3_tts_ref_audio /Users/horota/Downloads/psychopass.wav \
  --qwen3_tts_ref_text "ネットって、物を調理するための刃物とか、記録するための紙とか、そういうレベルのものじゃないですかね。いい悪いじゃない、そこにあるんだから受け入れる。" \
  --qwen3_tts_language ja
```

`--qwen3_tts_ref_text` は参照音声の書き起こしで、音声と一致していないと
その内容が出力の頭に混ざる。既定値は無関係な英文なので、必ず指定する。
Base は参照音声の音響プレフィックスがあるため短文でも正しく終端し、
`repetition_penalty` は既定の 1.05 のままでよい。

### 起動時の warmup

英文 `Hello, this is a warmup.` を日本語設定で読ませるため EOS が出ないことが
あるが、トークン予算の修正で上限が 360 トークン(28.8秒分)から 124 トークン
(9.9秒分)に下がった。モデル読み込みを含めた起動が実測 15秒 → **4.6秒**。

打ち込んだターンは `response.cancel` → `conversation.item.create` → `response.create`
の3イベントで送られる。サーバ側は元から対応していたので、変更は同梱クライアントだけ。
喋っている最中に送ると即座に割り込む(声で割り込んだときと同じ挙動)。

## 未解決

- **STT の幻聴（一部）** — 完全一致のブロックリストと2文字未満の除去は入った
  (`STT/hallucinations.py`)。**`ありがとうございました` 単体はまだ通る**(STATUS.md C-1)。
  単純にリストへ足すと正常な発話を捨てるので、発話長と VAD 信頼度を STT 段まで
  運ぶ配管が要る。L0-5(計測の拡張)で同じ配管を通す
- **キリル文字・ハングルの混入** — 未対応。`--stt parakeet-tdt` との比較が先
- **再生の途切れ** — 原因の切り分け中（`companion/phase0/FINDINGS-selfbargein.md`）
- **打ち込み中に VAD が発火すると衝突する** — マイクは開いたままなので、打鍵音や
  独り言で音声ターンが立つと `response.create` が
  `conversation_already_has_active_response` で弾かれる。打った文は履歴に入るが
  返事が来ない。`ERROR:` が1行出るだけ。直すなら入力行に文字がある間だけ
  マイク送信を止める（`callback_send` を入力バッファの状態でゲートする）
- **`--text-input` 時の Ctrl-C は既存の終了経路を通らない** — prompt_toolkit が
  raw mode で ISIG を落とすので SIGINT が飛ばず、`s2s_pipeline.py` の
  signal_handler が走らない。「終了しています」の表示が出ず、`ThreadManager.stop()`
  の5秒 join 上限ではなく `wait()` の無制限 join になる。実際には各ハンドラが
  0.1秒ポーリングで stop_event を見るので終了するが、詰まった段があると待ち続ける

## 解決済み

- ~~声質が毎回変わる~~ — `--qwen3_tts_gen_temperature 0` で greedy。
  `backend_registry` の `gen_` 接頭辞規約に乗るのでハンドラ側の配線は不要だった
- ~~日本語が文単位でストリーミングされない~~ — `LLM/japanese_segmenter.py`。
  NLTK は「。」で切らないので、言語コードで tokenizer を分岐させた
