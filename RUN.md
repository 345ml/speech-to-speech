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
  --qwen3_tts_model_name mlx-community/Qwen3-TTS-12Hz-1.7B-Base \
  --qwen3_tts_mlx_quantization 8bit \
  --qwen3_tts_gen_temperature 0 \
  --qwen3_tts_ref_audio /Users/horota/Downloads/psychopass.wav \
  --qwen3_tts_ref_text "ネットって、物を調理するための刃物とか、記録するための紙とか、そういうレベルのものじゃないですかね。いい悪いじゃない、そこにあるんだから受け入れる。" \
  --qwen3_tts_language ja
```

`--qwen3_tts_ref_text` は参照音声の書き起こしで、音声と一致していないと
その内容が出力の頭に混ざる。既定値は無関係な英文なので、必ず指定する。

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
