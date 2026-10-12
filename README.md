# CLM + SOUP: un agente di gioco che si migliora da solo

Progetto personale di studio. L'obiettivo è capire fin dove arriva un sistema che:

1. **decide velocemente** la mossa in un gioco, usando CLM come "System One";
2. **registra** partite, vittorie e sconfitte;
3. **si riaddestra** sui propri dati per giocare sempre meglio, usando la pipeline di
   fine-tuning di CLM e SOUP per la parte LLM.

Questo documento raccoglie la valutazione di fattibilità (ottobre 2026) e il piano di lavoro.

---

## 1. Cosa sono davvero i due strumenti

### CLM (Contrastive Language Models, Stanford + NVIDIA)

Repo: <https://github.com/Contrastive-LM/clm> · licenza Apache 2.0 · rilasciato a settembre 2026.

- **Non è un LLM generativo.** È un modello a due torri: un encoder **Qwen3-8B congelato**
  trasforma in vettori lo *stato* e ogni *azione candidata*; due piccole teste MLP
  (circa 20M parametri ciascuna, 4096 → 512 dimensioni) le proiettano in uno spazio comune.
  Il punteggio di un'azione è la similarità coseno con lo stato, e un softmax sui punteggi
  dà la distribuzione di probabilità.
- **Non ragiona e non genera testo.** Sceglie tra le opzioni che gli dai. Se l'opzione giusta
  non è nella lista, non la inventa.
- **Velocità.** ~28 ms per uno stato nuovo, ~0,7 ms per uno stato già visto (cache),
  misurati su una RTX 4090. Il costo per azione aggiuntiva è un prodotto scalare, quindi
  migliaia di candidati sono gestibili. Il README non indica un limite massimo di azioni.
- **Requisiti.** L'encoder gira con vLLM su GPU: i soli pesi di Qwen3-8B in bf16 occupano
  circa 16 GB, quindi una GPU da 24 GB è il caso comodo. Le teste girano anche su CPU.
- **Addestramento.** `train/finetune.py` allena **solo le teste**, con loss InfoNCE, su
  embedding precalcolati. Una volta calcolati gli embedding è un addestramento leggero.
  Attenzione: il trainer impara a imitare le coppie (stato, azione) che gli dai e **non usa
  il reward**. Le coppie vanno quindi filtrate prima: si danno solo le mosse buone.
- **Il demo di gioco (T-Rex)** va letto bene: un planner fisico etichetta ogni azione come
  "Safe/Unsafe" e CLM legge queste etichette. Buona parte dell'intelligenza sta nel planner,
  mentre CLM fa da selettore veloce. È un indizio importante per il nostro progetto: **conta
  molto come descriviamo lo stato e le azioni**.

### SOUP

Repo: <https://github.com/MakazhanAlpamys/Soup> · licenza Apache 2.0.

- È una CLI per il **fine-tuning di LLM generativi** configurata da un file YAML: SFT, DPO,
  GRPO, PPO, KTO, ORPO, LoRA/QLoRA, più task `embedding`, `classifier` e `reranker`.
- Il "layer streaming" (beta) promette il training di un 8B anche su una GPU da 4 GB.
- GRPO accetta una **funzione di reward custom** scritta in Python (`reward_fn: ./mio_reward.py`).
- Gli ambienti RL inclusi sono **single-shot**: prompt e risposta, senza un loop di partita
  multi-turno con il modello dentro. Il codice lo dice esplicitamente (`envs/guess_number.py`).
- Richiede Python 3.10–3.12. Su CPU funziona solo in modo sperimentale e molto lento.

## 2. Si possono unire?

**Non sullo stesso modello.** SOUP non allena le teste di CLM. Se si ri-addestrasse con SOUP
l'encoder Qwen3-8B, le teste pre-addestrate di CLM smetterebbero di funzionare, perché sono
legate a quell'encoder congelato.

**Sì, come due ruoli distinti** (System One / System Two):

| Ruolo | Strumento | Cosa fa |
|---|---|---|
| Giocatore veloce | CLM | Sceglie la mossa tra quelle legali, in pochi ms |
| Allenatore lento | LLM (es. Qwen3 1.7B/4B) addestrato con SOUP | Analizza le partite perse, interviene quando CLM è insicuro, genera esempi difficili |
| Arbitro | Motore di gioco in Python | Unica fonte di verità su mosse legali, vittorie e sconfitte |

## 3. Memorizzare tutto o generalizzare?

- **Tabella "stato → mossa migliore"**, cioè collezionare tutti i casi. Funziona solo in giochi
  piccoli come il tris. In Yu-Gi-Oh! lo stesso stato non si ripresenta praticamente mai (mano,
  campo, cimitero, informazioni nascoste), quindi la tabella resta quasi sempre vuota.
- **"Escludere le mosse che falliscono sempre"** è rischioso. Una mossa è buona o cattiva a
  seconda del contesto, e una sconfitta al turno 15 può dipendere da una mossa del turno 3:
  è il problema dell'assegnazione del merito (*credit assignment*).
- **Generalizzazione: la strada giusta, ed è esattamente ciò in cui CLM è forte.** Stati simili
  producono vettori vicini, quindi ciò che impara su un caso si trasferisce ai casi simili.
- La ricetta che funziona davvero (AlphaZero, Leela Zero) è **self-play + ricerca + rete**:
  una ricerca, per esempio MCTS o simulazioni, trova mosse migliori di quelle che la rete
  proporrebbe da sola. Si addestra la rete su quelle e si ripete il ciclo. Il segnale "ha vinto
  la ricerca" è molto più ricco del solo "ho vinto/perso la partita".

## 4. Architettura proposta

```
            ┌──────────────────────── ciclo di miglioramento ────────────────────────┐
            ▼                                                                        │
  Motore di gioco ──stato + mosse legali──► Cervello (pluggable)                     │
  (Python, regole)                          ├─ RandomBrain     (CPU, baseline)       │
       ▲                                    ├─ HeuristicBrain  (CPU, regole)         │
       │ mossa                              ├─ MiniCLMBrain    (CPU, encoder piccolo)│
       └────────────────────────────────────┤─ CLMBrain        (GPU, clm-serve)      │
                                            └─ LLMCoach        (SOUP, quando incerto)│
                                                       │                             │
                                     log JSONL: stato, mossa, esito, reward          │
                                                       ▼                             │
                         Data builder: filtra le mosse buone + negativi difficili    │
                                 ├─► formato CLM  ──► train/finetune.py ─────────────┤
                                 └─► formato SOUP (DPO/SFT) ──► soup train ──────────┘
                                                       │
                                                Valutazione: Elo / win-rate
                                                contro le versioni precedenti
```

## 5. Quale gioco

| Gioco | Pro | Contro | Verdetto |
|---|---|---|---|
| Yu-Gi-Oh! completo | Motivante | Regole enormi, informazioni nascoste, catene; serve un motore esterno (ygopro-core, vedi [sbl1996/ygo-agent](https://github.com/sbl1996/ygo-agent)) | Obiettivo finale, non il primo passo |
| **Mini-duello di carte** (stile Yu-Gi-Oh!, ~20 carte, regole semplificate) | Divertente, motore scritto da noi, testabile su CPU | Va progettato | **Consigliato per partire** |
| Forza 4 / Othello | Regole note, si può misurare la "mossa perfetta" | Meno originale | Ottimo come banco di prova |
| Briscola / Scopa | Informazione nascosta, poche mosse | Poche azioni, quindi CLM non dà il meglio | Opzionale |

## 6. Limiti da tenere a mente

- **Niente GPU = niente CLM vero.** Qwen3-8B non entra in RAM sul PC di lavoro né su un runner
  gratuito di GitHub Actions. Qui si costruisce e si testa tutto il resto.
- CLM è giovane: i benchmark sono dichiarati dagli autori e non ancora replicati da terzi.
- CLM imita e non esplora: l'esplorazione, cioè provare mosse nuove, va gestita da noi
  (epsilon-greedy, campionamento con temperatura, ricerca).
- Contesto di default di 2048 token: lo stato va descritto in modo compatto e in prosa, non in JSON.
- "Gratuito" è vero per il software. Il calcolo ha un costo: la GPU di casa, oppure le quote
  gratuite di Kaggle/Colab, che per Qwen3-8B sono al limite e vanno verificate.

## 7. Roadmap

**Fase 0: qui, solo CPU e GitHub**
- [ ] Motore del mini-duello in Python puro, con test.
- [ ] Interfaccia `Brain` e i cervelli Random/Heuristic.
- [ ] Self-play runner e logger JSONL nel formato transizioni di CLM
      (`state`, `action`, `task_id`, `step_idx`, `trajectory_id`, `reward`).
- [ ] Data builder che produce due formati: coppie positive filtrate per CLM, e coppie
      preferita/scartata per SOUP DPO.
- [ ] `MiniCLMBrain`: lo stesso schema a due torri con InfoNCE, ma con un encoder piccolo o
      con feature fatte a mano, per provare l'intero ciclo di apprendimento su CPU.
- [ ] Test su GitHub Actions, per verificare tutto senza eseguire nulla in locale.

**Fase 1: PC di casa con GPU**
- [ ] `vllm serve Qwen/Qwen3-8B --runner pooling` + `clm-serve`, poi `CLMBrain` collegato via HTTP.
- [ ] Baseline zero-shot di CLM contro Random/Heuristic.
- [ ] Primo ciclo: self-play → filtro → `train/finetune.py --task clm` → nuova testa
      (hot-reload) → valutazione.

**Fase 2: ricerca e allenatore**
- [ ] MCTS o simulazioni come "maestro" che genera mosse migliori delle proprie.
- [ ] LLM allenatore con SOUP (DPO sulle mosse migliori contro quelle peggiori), chiamato
      quando la confidenza di CLM è bassa.

## Note tecniche per le prossime sessioni

- Nel container cloud usato per la prima valutazione **huggingface.co era bloccato** dalla
  policy di rete, mentre PyPI e GitHub erano raggiungibili.
- Il container aveva Python 3.13, ma SOUP richiede 3.10–3.12: va usato un ambiente con 3.12.
- Il checkpoint delle teste di CLM è un `torch.save` con `state_head`, `action_head`,
  `logit_scale` e `cfg`. Si serve con `clm-serve --ckpt PATH` e si ricarica a caldo quando
  il file cambia.
