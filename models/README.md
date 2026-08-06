# Modelli opzionali

Il programma deve avviarsi anche senza modelli AI. I modelli vengono aggiunti progressivamente e rilevati automaticamente.

Struttura prevista:

```text
models/
  lama/lama.onnx
  codeformer/codeformer.pth
  3ddfa/mb1_120x120.onnx
  insightface/
  dfdnet/
  gfrnet/
```

I pesi non vengono inseriti automaticamente nel repository perché possono essere grandi e avere condizioni separate dal codice. La build finale includerà soltanto i modelli verificati e compatibili con Windows 11 x64.
