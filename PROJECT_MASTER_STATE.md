# Conservative Face Studio — stato operativo

Data: 2026-10-13. Branch unico: `integration/final-paper-quality-local`.

Mandato attuale: `docs/AUTONOMOUS_EXECUTION_MANDATE.txt`; prevale sul testo precedente copiato senza modifiche da `meta/project-state:PROJECT_EXECUTION_RULES.md` in `docs/PROJECT_EXECUTION_RULES.md`. Il registro operativo viene mantenuto su questo stesso branch; gli altri restano read-only.

## Fase 0 — inventario riconciliato

SHA di codice inventariato: `dca9b3534dbeffec11a7cee4b3dd8c449a5872d8`. Origine del registro storico: `meta/project-state@b8620b5f1e6d7e70608ce67448a8d506f7e11d4e`.

Branch e SHA completi: `config/current-session-inventory.json`. Il branch esistente discende da main `2767513f95dde2d417e7c6f1faf2357149a1a32f`; non viene ricreato né riscritto.

| Modello | Documentato | Codice / collegamento pipeline | Peso locale | Inferenza nella sessione |
|---|---|---|---|---|
| opencv_yunet | sì | Face/Landmarks; YuNet | no | NOT_RUN |
| opencv_sface | sì | Identity; SFace | no | NOT_RUN |
| nafnet_gopro_width32 | sì | NOT_ACTIVE; candidate/catalog only | no | NOT_RUN |
| nafnet_sidd_width32 | sì | NOT_ACTIVE; candidate/catalog only | no | NOT_RUN |
| restormer_motion_deblur | sì | NOT_ACTIVE; candidate/catalog only | no | NOT_RUN |
| restormer_real_denoise | sì | NOT_ACTIVE; candidate/catalog only | no | NOT_RUN |
| mirnet_fivek_enhance | sì | NOT_ACTIVE; candidate/catalog only | no | NOT_RUN |
| zero_dce_plus | sì | NOT_ACTIVE; candidate/catalog only | no | NOT_RUN |
| mediapipe_face_landmarker | sì | NOT_ACTIVE; candidate/catalog only | no | NOT_RUN |
| bisenet_face_parsing | sì | NOT_ACTIVE; candidate/catalog only | no | NOT_RUN |
| dmdnet | sì | NOT_ACTIVE; candidate/catalog only | no | NOT_RUN |
| lama_big | sì | NOT_ACTIVE; candidate/catalog only | no | NOT_RUN |
| 3ddfa_mb1 | sì | NOT_ACTIVE; candidate/catalog only | no | NOT_RUN |
| insightface_identity | sì | NOT_ACTIVE; candidate/catalog only | no | NOT_RUN |
| realesrgan_x2plus | sì | NOT_ACTIVE; candidate/catalog only | no | NOT_RUN |
| opencv_nafnet_deblur | sì | Deblur; NAFNet | no | NOT_RUN |
| face_parsing_resnet18_onnx | sì | semantic support | no | NOT_RUN |
| head_pose_mobilenetv2_onnx | sì | pose gate | no | NOT_RUN |
| opencv_lama_inpaint | sì | generated residual fallback | no | NOT_RUN |
| codeformer_v010 | sì | NOT_ACTIVE; candidate/catalog only | no | NOT_RUN |
| gfpgan_v13 | sì | NOT_ACTIVE; candidate/catalog only | no | NOT_RUN |
| restoreformer_v13_asset | sì | NOT_ACTIVE; candidate/catalog only | no | NOT_RUN |
| refstar_research | sì | NOT_ACTIVE; candidate/catalog only | no | NOT_RUN |
| instantrestore_research | sì | NOT_ACTIVE; candidate/catalog only | no | NOT_RUN |
| osdface_research | sì | NOT_ACTIVE; candidate/catalog only | no | NOT_RUN |
| fbcnn_color | sì | InstalledPaperQualityRuntime JPEG; development only | no | NOT_RUN |
| lraspp_damage_mask | sì | DamageMaskRuntime; rejected development artifact | no | NOT_RUN |
| deeplab_yunet_v2 | sì | research only; frozen gate FAIL | no | NOT_RUN |
| gfpgan_v14 | sì | research vertical slice only | no | NOT_RUN |
| gpen_bfr512 | sì | pinned upstream entry; no active installed route | no | NOT_RUN |
| restoreformer_plus_plus | sì | NOT_PRESENT | no | NOT_RUN |
| reffaceinpainting | sì | NOT_PRESENT | no | NOT_RUN |
| refinefir | sì | NOT_PRESENT | no | NOT_RUN |
| ref_ldm | sì | NOT_PRESENT | no | NOT_RUN |

### Codice inattivo, no-op e divergenze

- Cataloghi e prototipi non provano caricamento o inferenza: MediaPipe restituisce solo 5 punti; 3DDFA non è collegato; i modelli avanzati sopra restano candidati.
- Enhance `blend=0.0` è già intercettato da `automatic_quality_policy`: blend effettivo 0.12 senza reference, ABSTAIN con reference. Il difetto riportato nell’inventario iniziale è storico; non è un no-op attuale nel percorso automatico.
- `reliability_aware_identity_flags` ammette un riferimento sotto 0.363 tramite un cluster: incompatibile con la verifica diretta richiesta. Correzione necessaria nella fase 1.
- `product-audit.json` e RELEASE_STATE descrivono candidati precedenti: non attestano questo HEAD.
- Il rilevatore DeepLab ha fallito il gate DEV storico: precisione 0.492098, recall 0.451897, F1 sticker 0.683359, scribble 0.271144. Il successo della CI riguarda l’esecuzione.
- Nessun peso reale è stato eseguito in questa sessione; nessun risultato fisico EliteBook, installer finale o Target95 attuale.

### Test e avanzamento

Suite iniziale: 1745 PASS, 2 FAIL, 2 SKIPPED. I due FAIL sono causati dalla home Linux read-only; riproduzione conservata in `/tmp/cfs-baseline-pytest.log`, correzione ambiente tramite XDG_DATA_HOME scrivibile (nessun test disabilitato). Riesecuzione completa con ambiente scrivibile: **1747 PASS, 0 FAIL, 2 SKIPPED in 23.45 s**; pip check PASS. Le due skip opzionali restano visibili.

CI nuova: Integration regression, Linux + Windows, suite intera e artifact JUnit; nessun workflow di certificazione o holdout viene invocato.

Fase 0: inventario completato, gate test PASS / CI PENDING. Fase 1: integrazioni selettive storiche già presenti (Track A → protocollo → Track B); nuova verifica di conformità PENDING. Fase 2: wiring storico, gate reale nella sessione PENDING. Fase 3: geometria densa non qualificata. Fasi 4–11: non chiuse secondo il mandato attuale.

Percentuale reale del prodotto finale: NON_MISURATA; il precedente 43/100 è una stima storica, non una metrica verificata. Fasi del mandato chiuse nella sessione: 0/12 finché test e CI non passano.

RELEASE_READY=FALSE; PROJECT_FINISHED=FALSE; V3/V4 consumati e intatti; V5 non creato; PR #2 non unita.

PROSSIMO PASSO ESATTO: chiudere fase 0 con suite e CI verdi; eliminare l’autorità d’identità transitiva nella fase 1 e rieseguire regressioni mirate e suite completa.

CI d3e1668: run 38406787876 FAIL prima dei job. Causa: contesto `runner.temp` non disponibile in jobs.env. Correzione minima: `github.workspace` per la directory dati dei test. Nessun test o soglia cambiati. CI corretta PENDING.

## Fase 0 — gate chiuso

Commit inventario `d3e1668cb08cdfa0798131395baf3e35f906fe30`; correzione CI `53bde8901f5c4f7a479ba1d20f95841beae41140`.
CI [38407104002](https://github.com/xhinoo97-svg/ConservativeFaceStudio/actions/runs/38407104002): **SUCCESS**, Linux + Windows, suite completa. Fase 0 PASS.

Sei pesi production ora presenti nel checkout ignorato da Git: hash verificati 6/6; smoke con inferenza reale CPU 6/6 PASS. Fixture sintetiche: nessuna metrica di restauro o verifica biometrica reale viene dedotta. Report: `config/session-model-smoke.json`.

## Fase 1 — conformità dell'integrazione

Le integrazioni precedenti rimangono selettive: Track A → `f558ac8`, `21f3a3d`, `7fe71d6`, `e5ea08a`; protocollo generico → `871556b`, `b05e5aa`; Track B → component bank `e7ce058`, damage runtime `f9c539c`, reference-first `0ed9c52`, risorse `ef30cde`, fusion `0d35f89`, timeline `b653a35`, FBCNN `6ea2871`. Origini: SHA dei tre branch riportati nell'inventario. Non vengono ripetuti merge o riscritti branch storici.

Correzione attuale: `app/pretrained_face_handlers.py` richiede score SFace diretto MAIN/reference, finito e compreso fra **0.363 e 1.0**. Un cluster non può promuovere referenze sotto soglia. Il precedente test che autorizzava tale promozione viene sostituito con il contratto più restrittivo del mandato; nuove regressioni per NaN, infinito, score fuori intervallo e soglia inclusiva. Conflitti: nessuno. Decisione: dati preflight consentiti per ranking, mai per autorità biometrica transitiva.

Test mirati: **22 PASS**. Suite completa: **1755 PASS, 0 FAIL, 2 SKIPPED**, 26.16 s. CI del nuovo commit PENDING. Nessuna soglia abbassata, modello promosso o certificazione eseguita.

PROSSIMO PASSO ESATTO: verificare CI fase 1, poi smoke end-to-end MAIN+0/1/3 reference con pesi reali; non chiudere fasi successive con sola evidenza di wiring.
