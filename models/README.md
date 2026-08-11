# Model store

L'installer offline contiene i sei checkpoint production verificati. I modelli opzionali non vengono attivati senza checksum, handler e smoke test reali.

Struttura prevista:

```text
models/
  detection/
  identity/
  landmarks/
  parsing/
  pose/
  deblur/
  reference/
  inpainting/
  restoration/
  optional/
```

I pesi non sono versionati in Git. La build scarica con HTTPS, limite di dimensione e SHA-256, esegue inferenza reale, esporta `model-registry.json` e stage soltanto i modelli production verificati. L'updater conserva la versione precedente e ripristina l'intero pack se un nuovo checkpoint fallisce.
