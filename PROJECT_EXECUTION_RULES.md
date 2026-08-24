La forma corretta è un unico installer scaricabile, con più modelli specialistici interni caricati uno alla volta. Un solo modello neurale non può essere il migliore contemporaneamente per identità, blur, sticker, allineamento e ricostruzione.

CONSERVATIVE FACE STUDIO
ESECUZIONE AUTONOMA FINALE — INTEGRAZIONE, PAPER QUALITY, TEST E RELEASE

Agisci come responsabile tecnico completo del progetto Conservative Face Studio: software architect, ML engineer, computer-vision researcher, Windows release engineer, tester e auditor.

Non limitarti ad analizzare o proporre un piano. Lavora concretamente sul repository: leggi il codice, individua le cause reali degli errori, implementa le correzioni, integra i modelli, esegui test, benchmark e training quando necessario, aggiorna i documenti, esegui commit e push, controlla le CI e continua cronologicamente fino a ottenere il massimo prodotto realmente verificato.

Lavora senza chiedere conferme per decisioni tecniche reversibili comprese in questo mandato. Fermati solo davanti a un blocco reale di permessi, licenza, credenziali, hardware non disponibile o azione irreversibile non autorizzata.

## 1. Risultato finale obbligatorio

Realizzare un unico prodotto scaricabile:

- una sola applicazione Conservative Face Studio;
- un solo installer Windows;
- un solo model pack offline;
- una sola configurazione finale;
- una sola pipeline automatica;
- nessuna versione duplicata proposta all’utente;
- funzionamento locale e offline;
- compatibilità con HP EliteBook 1030/310 G3, Windows, 16 GB RAM, CPU Intel, senza dipendenza obbligatoria da CUDA;
- possibilità di usare modelli leggeri e modelli avanzati, caricandone uno pesante alla volta;
- interfaccia reattiva durante tutte le elaborazioni;
- nessun errore runtime nella matrice finale di accettazione.

Il pacchetto unico può contenere diversi modelli specializzati. Non forzare un solo network a eseguire compiti per cui non è adatto. L’utente deve comunque scaricare e installare un solo prodotto.

## 2. Stato iniziale da verificare

Prima di modificare qualsiasi cosa, controlla lo stato remoto effettivo:

- repository: `xhinoo97-svg/ConservativeFaceStudio`;
- `main`: baseline certificata V1;
- `main` noto: `2767513f95dde2d417e7c6f1faf2357149a1a32f`;
- Track A: `hotfix/real-world-restoration-v1.1`;
- Track A HEAD noto: `77687b3b171f4e9989fcf486834f2d8b7a52f591`;
- candidato Track A valutato: `b6ce7ebde87d4ce84e5849664716dc3e822ad762`;
- Track B: `research/paper-quality-local-v2`;
- Track B HEAD noto: `6d57725aae087bb4a3144d521d91346999f9a4fd`;
- protocollo futuro: `protocol/v5-certification-hardening`;
- protocollo HEAD noto: `268188c5a2540455ff804383cb583b16546b62f1`;
- registro canonico: `meta/project-state`;
- registro HEAD noto: `365af5ea54ba25960657ba85065c0cab43da9a8b`;
- PR #2: OPEN, DRAFT, NO-GO;
- V3: consumato;
- V4: `CONSUMED_FAIL`, 0/40 casi eseguiti;
- Paper Quality: non qualificato;
- Target95: non raggiunto;
- test fisico sull’EliteBook: non eseguito.

Se gli SHA sono cambiati, usa lo stato remoto più recente e documenta la differenza. GitHub, commit, artifact, log e risultati riproducibili hanno autorità superiore ai messaggi precedenti.

Non modificare o cancellare le evidenze V3/V4. Non rieseguire holdout consumati. Non unire PR #2 come candidato certificato.

## 3. Definizione del prodotto

L’applicazione deve accettare:

- una MAIN IMAGE;
- da zero a nove fotografie di riferimento;
- reference complete o parziali;
- reference con pose, espressioni, luci e risoluzioni differenti;
- reference dove è visibile solamente un componente utile.

Deve riconoscere e gestire:

- identità della persona;
- volto singolo o riferimento senza volto;
- reference della persona sbagliata;
- posa frontale, profilo, tre quarti, yaw, pitch e roll;
- occhio destro e sinistro;
- sopracciglio destro e sinistro;
- naso;
- bocca e labbra;
- sorriso ed espressione;
- mento;
- guance;
- mascella;
- fronte;
- contorno del volto;
- blur globale;
- blur locale;
- motion blur;
- defocus blur;
- pixelazione;
- mosaico;
- artefatti JPEG e doppio JPEG;
- sticker;
- emoji;
- scarabocchi;
- linee colorate;
- black bar;
- blocco opaco;
- occlusione parziale;
- componente mancante;
- danni misti.

## 4. Regole immutabili di identità e provenienza

La MAIN è l’autorità per:

- posa;
- espressione;
- sorriso;
- composizione;
- proporzioni;
- geometria;
- sfondo;
- canvas finale.

Le reference sono fonti di identità e dettaglio. Non devono trasformare la MAIN nella posa o nell’espressione della reference.

Ordine obbligatorio delle fonti:

1. `MAIN_OBSERVED`;
2. `OBSERVED_REFERENCE`;
3. `SYMMETRY_INFERRED`;
4. `GENERATED_MODEL_INFERRED`;
5. `UNRESOLVED`.

Regole obbligatorie:

- wrong-person final pixels: `0`;
- provenance violations: `0`;
- nessuna reference sbagliata può diventare donatore;
- nessun proxy o istogramma può sostituire una verifica biometrica;
- SFace `0.363` non deve essere abbassato per superare i test;
- una verifica d’identità non può essere propagata transitivamente;
- non modificare le parti sane senza un miglioramento dimostrato;
- rispettare il limite MAE esterno alla zona autorizzata dove previsto;
- non classificare pixel generati come osservati;
- non utilizzare un risultato più bello ma appartenente a una persona diversa;
- non cambiare artificialmente naso, bocca, occhi, mento o sorriso;
- conservare il dettaglio originale sano;
- se l’informazione reale non esiste, usare astensione, rollback o `UNRESOLVED`.

Se un naso, un occhio o una bocca sono completamente distrutti e non sono visibili in nessuna reference, il programma non può dichiarare ricostruzione reale al 95%. Può generare una proposta fotorealistica, ma deve marcarla `GENERATED_MODEL_INFERRED` e assegnare una confidenza inferiore.

## 5. Ordine cronologico obbligatorio

Non saltare le fasi. Ogni fase deve avere un gate verificabile prima della successiva.

### Fase 0 — Inventario e verità del repository

1. Leggi `PROJECT_MASTER_STATE.md`, `RELEASE_STATE.md`, documenti architetturali, model registry, workflow, installer e test.
2. Elenca branch, SHA, divergenze e file modificati.
3. Verifica quali modelli sono realmente usati dalla pipeline.
4. Distingui:
   - modello presente nei documenti;
   - modello presente nel codice;
   - modello collegato alla pipeline;
   - modello con peso reale;
   - modello incluso nell’installer;
   - modello realmente testato.
5. Trova codice morto, prototipi isolati, blocchi no-op e indicatori UI non veritieri.
6. Aggiorna il registro canonico con lo stato misurato.

Gate: nessuna affermazione sul progetto senza prova nel codice, nei test o negli artifact.

### Fase 1 — Un solo branch di integrazione

Crea un unico branch attivo:

`integration/final-paper-quality-local`

Parti da `main`.

Integra in ordine:

1. correzioni operative valide del candidato Track A;
2. escludi request e marker V4 dalla nuova linea di sviluppo;
3. integra il protocollo generico verificato;
4. porta i moduli Track B uno alla volta;
5. risolvi i conflitti con test dopo ogni gruppo;
6. conserva i branch precedenti come storia tecnica read-only.

Non eseguire un merge wholesale cieco. Registra per ogni integrazione:

- commit di origine;
- file integrati;
- conflitti;
- decisione;
- test;
- risultato.

Gate: full test suite verde sul nuovo branch prima di aggiungere nuovi modelli.

### Fase 2 — Riparazione della pipeline esistente

Controlla tutti i 13 blocchi:

1. Import;
2. Deblur;
3. Enhance;
4. Face/Landmarks;
5. Align References;
6. Detect Damage;
7. Select Best Regions;
8. Repair/Inpaint;
9. Fusion;
10. Pose;
11. Identity Check;
12. Upscale;
13. Export.

Per ogni blocco verifica:

- input reale;
- output reale;
- modello eseguito;
- peso caricato;
- errore gestito;
- rollback;
- provenienza;
- consumo CPU/RAM;
- timer;
- test unitario;
- test di integrazione.

Elimina o correggi il no-op `Enhance blend=0.0`. Se non viene eseguito alcun modello, mostra `SKIPPED`; non visualizzare “JPEG specialist” quando FBCNN non è stato caricato.

Gate: ogni blocco deve funzionare separatamente prima del test end-to-end.

### Fase 3 — Geometria facciale densa

Mantieni YuNet e SFace per rilevamento e identità, ma non usare soltanto i cinque landmark per componenti complessi.

Qualifica:

- MediaPipe Face Landmarker, 478 landmark 3D;
- 3DDFA\_V2, geometria 3D e percorso ONNX CPU;
- eventuale alternativa con licenza e pesi compatibili.

Confronta:

- accuratezza occhi;
- contorno labbra;
- naso;
- mento;
- profilo;
- pose estreme;
- volto parzialmente coperto;
- CPU, RAM e durata.

Crea una mappa canonica del volto e allinea ogni reference alla geometria della MAIN senza frontalizzare obbligatoriamente l’immagine finale.

Gate: allineamento componente-per-componente verificato su pose diverse e reference parziali.

### Fase 4 — Rilevamento preciso del danno

La LR-ASPP attuale è solo una baseline DEV. Non integrarla in produzione finché non supera qualità, dominio, licenza e Windows.

Costruisci un damage detector multi-classe per:

- healthy;
- blur;
- motion blur;
- pixelation;
- block mosaic;
- JPEG;
- scribble;
- sticker;
- opaque block;
- black bar;
- partial occlusion;
- missing component.

Usa un’architettura compatibile con CPU, per esempio LR-ASPP/MobileNetV3 o DeepLab leggero, con pesi redistribuibili.

Priorità della maschera:

- alta precisione per non rimuovere pixel sani;
- recall sufficiente per non lasciare residui;
- accuratezza dei bordi;
- separazione tra sticker e parti reali del viso;
- separazione tra blur locale e immagine semplicemente poco nitida.

Gate DEV minimo:

- precisione binaria `>=0,95`;
- recall binaria `>=0,90`;
- sticker F1 `>=0,90`;
- scribble F1 `>=0,90`;
- motion/local blur F1 `>=0,85`;
- nessuna classe critica con F1 zero;
- validazione per sesso, tonalità della pelle, età adulta e posa.

### Fase 5 — Banca personalizzata per componenti

Integra realmente nella pipeline:

- `PersonalizedReferenceBank`;
- `PersonalizedComponentSelector`;
- reference-first repair;
- component-aware fusion;
- provenienza per pixel.

Per ogni reference:

1. rileva il viso;
2. verifica la stessa identità;
3. stima posa e qualità;
4. rileva danni;
5. misura visibilità di ogni componente;
6. allinea la reference alla MAIN;
7. salva solo evidenza valida.

Crea banche separate per:

- occhio destro;
- occhio sinistro;
- sopracciglio destro;
- sopracciglio sinistro;
- naso;
- philtrum;
- bocca/labbra;
- guancia destra;
- guancia sinistra;
- mento;
- mascella;
- fronte;
- contorno.

Il naso può provenire dalla reference 2, l’occhio dalla reference 5 e la bocca dalla reference 1. Non scegliere obbligatoriamente un’unica “migliore reference” per tutto il volto.

Gate: test end-to-end in cui almeno tre componenti provengono da reference differenti e la MAIN conserva posa ed espressione.

### Fase 6 — Modelli pretrained e modelli avanzati

Non limitarti ai modelli leggeri. Prova anche modelli complessi, purché possano essere eseguiti, convertiti o adattati al PC finale.

Qualifica almeno:

- NAFNet: deblur leggero;
- Restormer: motion blur, defocus e restauro avanzato;
- FBCNN: JPEG e doppio JPEG;
- GPEN;
- GFPGAN;
- CodeFormer;
- RestoreFormer++;
- RefineFIR: una reference e dettagli fini;
- RefFaceInpainting: grande occlusione con reference;
- DMDNet: banca occhi/naso/bocca e reference variabili;
- InstantRestore: restauro personalizzato con più reference;
- ReF-LDM: numero flessibile di reference;
- FaceMe o alternativa equivalente per identità personalizzata.

Per ogni modello registra:

- repository ufficiale;
- paper;
- commit esatto;
- licenza del codice;
- licenza dei pesi;
- URL ufficiale del peso;
- SHA-256;
- dimensione;
- preprocessing;
- postprocessing;
- dipendenze;
- input/output;
- CPU;
- RAM;
- tempo;
- qualità;
- identità;
- Windows;
- offline;
- possibilità di redistribuzione.

Ordine delle fonti online:

1. paper ufficiale;
2. repository ufficiale;
3. documentazione ufficiale;
4. release e model card;
5. issue ufficiali;
6. forum tecnici come pista diagnostica;
7. verifica locale obbligatoria prima di usare qualsiasi soluzione trovata nei forum.

Non copiare codice casuale da forum. Una soluzione trovata online deve essere compresa, verificata, adattata e coperta da test.

### Fase 7 — Ottimizzazione dei modelli pesanti

Se un modello avanzato supera la qualità ma è troppo pesante:

1. esegui solo sul crop facciale;
2. riduci la risoluzione interna mantenendo export ad alta qualità;
3. usa tiling;
4. prova ONNX Runtime;
5. prova OpenVINO per Intel;
6. prova DirectML se realmente utile;
7. prova FP16 solo se supportato;
8. prova INT8 con calibrazione;
9. prova pruning o distillazione;
10. scarica il modello dalla RAM dopo il blocco.

Non è sufficiente “prendere solo una parte del codice”. Modello, peso, architettura e preprocessing devono restare compatibili. Ogni conversione deve superare una verifica di parità rispetto al modello ufficiale.

Il prodotto può avere due modalità nello stesso installer:

- `LOCAL STANDARD`: più rapida;
- `PAPER QUALITY`: più lenta e con modelli complessi.

Entrambe devono essere locali. Paper Quality può richiedere più tempo, ma non deve bloccarsi o superare i limiti hardware.

Gate:

- RAM di processo `<80%`;
- RAM totale `<80%`;
- CPU governata `<80%` quando tecnicamente possibile;
- un solo modello pesante caricato;
- nessun crash, blocco UI o memory leak.

### Fase 8 — Dominio femminile

Il prodotto deve essere particolarmente accurato sui volti femminili adulti.

Non cercare necessariamente un modello dichiarato “female-only”. Prima misura i modelli generali sul dominio femminile. Se esiste un gap reale:

1. costruisci un dataset femminile adulto con licenza/consenso;
2. includi diverse tonalità della pelle;
3. includi età adulte differenti;
4. includi trucco, assenza di trucco, occhiali e capelli parzialmente coprenti;
5. includi pose ed espressioni;
6. esegui fine-tuning o adapter sul modello migliore;
7. confronta il modello generale e quello adattato;
8. conserva la variante femminile solamente se produce un miglioramento misurabile senza degradare identità e sicurezza.

Non usare fotografie di minori. Non scaricare casualmente immagini da social network, forum o motori di ricerca.

### Fase 9 — Dataset e training

Le prime 100–200 fotografie possono essere utilizzate come pilot, ma non bastano per certificare il 95%.

Costruisci:

- TRAIN;
- DEV;
- VALIDATION;
- FINAL\_HOLDOUT.

Gli split devono essere identity-disjoint.

Per ogni immagine registra:

- fonte;
- licenza o consenso;
- data;
- hash;
- identità interna pseudonimizzata;
- sesso/dominio;
- età adulta approssimativa;
- posa;
- luce;
- risoluzione;
- danno;
- severità;
- maschera;
- seed;
- relazione MAIN/reference;
- split.

Genera migliaia di degradazioni riproducibili partendo da fotografie lecite:

- blur di tipi diversi;
- sticker realistici;
- emoji;
- scarabocchi;
- mosaico;
- JPEG;
- occlusioni;
- danni misti.

Aggiungi anche fotografie realmente degradate, perché il solo danno sintetico non dimostra generalizzazione reale.

Training:

1. usa pretrained quando disponibile;
2. fine-tuning solo su TRAIN;
3. selezione e soglie solo su DEV;
4. una verifica congelata su VALIDATION;
5. nessun tuning sul FINAL\_HOLDOUT;
6. dopo tre ipotesi fallite, interrompi quella famiglia e cambia modello, dati o strategia.

### Fase 10 — Matrice MAIN + 0–9 reference

Testa separatamente:

- MAIN senza reference;
- MAIN + 1 reference;
- MAIN + 2;
- MAIN + 3;
- MAIN + 4;
- MAIN + 5;
- MAIN + 6;
- MAIN + 7;
- MAIN + 8;
- MAIN + 9.

Per ogni quantità testa:

- danni singoli;
- danni misti;
- tre severità;
- pose differenti;
- reference utili;
- reference parziali;
- reference sfocate;
- reference senza componente necessario;
- reference con luce diversa;
- reference senza volto;
- reference wrong-person;
- più volti nella stessa immagine;
- riferimento duplicato;
- file corrotto;
- formati e risoluzioni differenti.

Test speciale obbligatorio:

- naso dalla reference A;
- occhio destro dalla reference B;
- occhio sinistro dalla reference C;
- bocca dalla reference D;
- MAIN con posa ed espressione preservate.

### Fase 11 — Definizione corretta del 95%

Definisci:

- `eligible_case`: danno reversibile oppure dettaglio vero presente nella MAIN o in una reference verificata;
- `accepted_case`: output non rollback e non abstention;
- `successful_case`: supera identità, danno, geometria, provenienza e qualità;
- `accepted_success_rate = successful_case / accepted_case`;
- `coverage = accepted_case / eligible_case`;
- `total_success_rate = successful_case / tutti_i_casi`.

Target:

- accepted success rate `>=95%`;
- nessun sottogruppo critico `<90%`;
- wrong-person pixels `0`;
- provenance violations `0`;
- runtime errors `0`;
- healthy-region damage `0` oltre le tolleranze congelate;
- report separato per 0–9 reference;
- report separato per sticker, scribble, blur, JPEG e componenti;
- intervallo di confidenza riportato;
- astensioni e rollback non possono gonfiare il 95%.

Non dichiarare 95% quando il risultato è solamente gradevole. La fedeltà deve essere riferita alla stessa persona e al dettaglio verificabile.

### Fase 12 — UI, loading e timeline

Implementa una timeline reale di 13 blocchi.

Mostra:

- blocco corrente;
- `x/13`;
- modello esatto caricato;
- peso e versione;
- stato di caricamento;
- tempo trascorso del blocco;
- ETA del blocco;
- tempo totale trascorso;
- ETA totale;
- percentuale totale ponderata sui tempi reali;
- CPU;
- RAM;
- componenti analizzati;
- reference selezionata per componente;
- `PASS`;
- `SKIPPED`;
- `ROLLBACK`;
- `ABSTAIN`;
- `ERROR`.

La UI deve:

- restare reattiva;
- consentire annullamento sicuro;
- scaricare i modelli;
- chiudersi senza corrompere il progetto;
- conservare log e stato;
- non mostrare un modello non eseguito;
- non mostrare percentuali inventate;
- imparare ETA dai tempi storici locali senza salvare fotografie o dati biometrici.

### Fase 13 — Gestione autonoma degli errori

Per ogni errore:

1. conserva log completo;
2. registra branch e SHA;
3. crea una riproduzione minima;
4. classifica la causa:
   - codice;
   - dipendenza;
   - modello;
   - peso;
   - hash;
   - preprocessing;
   - postprocessing;
   - memoria;
   - Windows;
   - CI;
   - dati;
   - licenza;
   - rete;
5. consulta fonti online nell’ordine stabilito;
6. identifica la root cause;
7. applica la correzione minima;
8. aggiungi un test di regressione;
9. esegui test mirati;
10. esegui full suite;
11. esegui CI;
12. aggiorna il registro.

I retry sono permessi in DEV e VALIDATION per errori infrastrutturali documentati. Non sono permessi sui holdout consumati.

Non risolvere un errore:

- disabilitando il test;
- cancellando il caso;
- abbassando la soglia;
- trasformando un errore in PASS;
- sostituendo un modello reale con un mock;
- marcando un peso non verificato come ACTIVE;
- ignorando una licenza incompatibile.

### Fase 14 — Test Windows e PC reale

Dopo i test Linux/CI:

1. crea model pack offline;
2. verifica tutti gli SHA-256;
3. costruisci portable package;
4. costruisci installer;
5. installa su Windows pulito;
6. disattiva la rete;
7. avvia l’applicazione installata;
8. esegui smoke test di ogni modello;
9. esegui casi MAIN + 0–9 reference;
10. controlla CPU, RAM, durata e UI;
11. esegui test sul vero EliteBook;
12. prova riavvio, disinstallazione e reinstallazione;
13. verifica che non manchino DLL, runtime, pesi o file.

Nessun modello può essere `ACTIVE` o `FALLBACK` se non supera il test dall’applicazione installata e offline.

### Fase 15 — Certificazione indipendente

V3 e V4 restano consumati.

Crea un V5 nuovo, identity-disjoint e mai osservato solamente dopo che:

- full suite è verde;
- pipeline completa è integrata;
- modelli sono qualificati;
- target95 è raggiunto su VALIDATION;
- Windows pulito passa;
- installer offline passa;
- EliteBook fisico passa;
- pesi e licenze sono verificati.

Prima di STARTED esegui:

- verifica del runner generico;
- fixture sintetiche;
- firma corretta di `build_freeze`;
- controllo ordine preflight/consumo;
- verifica del candidato;
- verifica dei pesi;
- verifica degli artifact precedenti.

Questo mandato autorizza una sola esecuzione V5 quando tutti i prerequisiti machine-readable risultano verdi. Se un prerequisito non è verde, non consumare V5.

Dopo STARTED:

- qualsiasi fallimento è terminale;
- nessun retry;
- nessun rerun;
- nessun tuning;
- conserva tutte le evidenze.

### Fase 16 — Prodotto finale unico

Solo se tutti i gate passano:

1. congela codice;
2. congela configurazione;
3. congela modelli e hash;
4. genera un solo model pack;
5. genera un solo installer;
6. genera un solo manuale;
7. genera un solo report finale;
8. crea una sola release candidate;
9. archivia le versioni precedenti come storia tecnica;
10. non presentare all’utente più versioni equivalenti.

Non unire automaticamente `main` se la regola del repository richiede revisione. Il prodotto finale deve comunque essere disponibile come installer verificato e release candidate unica.

## 6. Aggiornamento del registro

Dopo ogni modifica significativa:

1. test;
2. artifact;
3. commit;
4. push;
5. verifica SHA remoto;
6. aggiornamento di `PROJECT_MASTER_STATE.md`.

Ogni report deve contenere:

- data;
- branch;
- SHA;
- obiettivo della fase;
- file modificati;
- modello e peso;
- hash;
- licenza;
- test eseguiti;
- test passati/falliti;
- dataset;
- metriche;
- errori;
- rollback;
- astensioni;
- wrong-person pixels;
- provenance violations;
- CPU;
- RAM;
- durata;
- Windows;
- EliteBook;
- percentuale reale di completamento;
- prossimo passo esatto.

## 7. Condizioni per dichiarare completamento

Puoi dichiarare `PROJECT_FINISHED=TRUE` solamente quando:

- esiste un solo installer scaricabile;
- installazione pulita riuscita;
- funzionamento offline riuscito;
- EliteBook riuscito;
- MAIN + 0–9 reference riuscito;
- sticker e scarabocchi rimossi nei casi eleggibili;
- blur trattato dal modello corretto;
- banca componenti realmente utilizzata;
- reference wrong-person bloccate;
- zero errori nella matrice finale;
- wrong-person pixels zero;
- provenienza violazioni zero;
- Target95 misurato e raggiunto;
- V5 superato;
- artifact, hash, log e report disponibili.

Non dichiarare “perfetto”, “senza errori”, “paper quality”, “95%” o “finito” sulla base di test unitari, demo, mock, un’unica identità, immagini selezionate manualmente o risultati dei paper.

Il risultato finale deve essere una fotografia naturale e fedele quando esiste informazione sufficiente. Nei casi impossibili deve essere onesto, conservativo e chiaramente classificato, senza inventare identità.