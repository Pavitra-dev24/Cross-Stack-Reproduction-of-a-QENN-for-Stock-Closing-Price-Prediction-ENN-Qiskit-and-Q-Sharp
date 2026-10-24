export const DATASETS = [
  'BSE',
  'NASDAQ',
  'HSI',
  'SSE',
  'Russell2000',
  'TAIEX',
] as const

export type DatasetName = (typeof DATASETS)[number]
export type Family = 'Classical' | 'Quantum' | 'Paper'

export interface ModelSeries {
  name: string
  family: Family
  hiddenNeurons?: number
  qubits?: number
  meanNmse: Record<DatasetName, number>
  stdNmse?: Partial<Record<DatasetName, number>>
  stdSource?: string
}

export interface GapRow {
  model: 'QENN-Qiskit' | 'QENN-QSharp'
  dataset: DatasetName
  paperQenn: number
  reproduced: number
  absGap: number
  relGapPct: number
}

export interface RuntimeRow {
  model: string
  epochs: number
  totalTrainSec: number
  secPerEpoch: number
  qSharpFallback: string
}

export interface ProtocolControl {
  parameter: string
  value: string
  impact: string
}

export const MODEL_SERIES: ModelSeries[] = [
  {
    name: 'ENN, 10 hidden neurons',
    family: 'Classical',
    hiddenNeurons: 10,
    meanNmse: {
      BSE: 0.06904915263876318,
      NASDAQ: 0.13192527554929256,
      HSI: 0.01510022757574916,
      SSE: 0.010147489025257528,
      Russell2000: 0.07832099548541009,
      TAIEX: 0.16227025222033262,
    },
    stdNmse: {
      BSE: 0.05483508493937339,
      NASDAQ: 0.16366949317696367,
      HSI: 0.008622003492922272,
      SSE: 0.006974949708158687,
      Russell2000: 0.07030574552006301,
      TAIEX: 0.19707998076635677,
    },
    stdSource: '10-seed full rerun (100 epochs)',
  },
  {
    name: 'ENN, 20 hidden neurons',
    family: 'Classical',
    hiddenNeurons: 20,
    meanNmse: {
      BSE: 0.015821479307487608,
      NASDAQ: 0.0244937248993665,
      HSI: 0.010472220648080111,
      SSE: 0.007420697901397944,
      Russell2000: 0.017639879090711474,
      TAIEX: 0.026033433713018896,
    },
    stdNmse: {
      BSE: 0.01460965205725666,
      NASDAQ: 0.020286777991341473,
      HSI: 0.0029633711294504224,
      SSE: 0.0058006814789915255,
      Russell2000: 0.022586643862032785,
      TAIEX: 0.016300560018936816,
    },
    stdSource: '10-seed full rerun (100 epochs)',
  },
  {
    name: 'ENN, 40 hidden neurons',
    family: 'Classical',
    hiddenNeurons: 40,
    meanNmse: {
      BSE: 0.01432454886380583,
      NASDAQ: 0.022880662651732563,
      HSI: 0.012902791239321231,
      SSE: 0.006932464335113764,
      Russell2000: 0.010435030143707991,
      TAIEX: 0.025545340962707996,
    },
    stdNmse: {
      BSE: 0.010440514035049451,
      NASDAQ: 0.01624234334500502,
      HSI: 0.005566796182791964,
      SSE: 0.0031370364599359246,
      Russell2000: 0.005332636278984538,
      TAIEX: 0.016708013396471377,
    },
    stdSource: '10-seed full rerun (100 epochs)',
  },
  {
    name: 'ENN, 70 hidden neurons',
    family: 'Classical',
    hiddenNeurons: 70,
    meanNmse: {
      BSE: 0.01452671056613326,
      NASDAQ: 0.02537587732076645,
      HSI: 0.010499943932518362,
      SSE: 0.005360616696998477,
      Russell2000: 0.01054257161449641,
      TAIEX: 0.024975603772327305,
    },
    stdNmse: {
      BSE: 0.016276580407660998,
      NASDAQ: 0.031633841960582824,
      HSI: 0.004435936838024147,
      SSE: 0.002112642729535589,
      Russell2000: 0.007678112618012511,
      TAIEX: 0.02368550459730481,
    },
    stdSource: '10-seed full rerun (100 epochs)',
  },
  {
    name: 'ENN, 100 hidden neurons',
    family: 'Classical',
    hiddenNeurons: 100,
    meanNmse: {
      BSE: 0.00915532438084483,
      NASDAQ: 0.012940835626795887,
      HSI: 0.009627329418435693,
      SSE: 0.004584120400249958,
      Russell2000: 0.007806525146588683,
      TAIEX: 0.021352891437709333,
    },
    stdNmse: {
      BSE: 0.005520611683085687,
      NASDAQ: 0.011520849903456875,
      HSI: 0.0036835083763173647,
      SSE: 0.0023486727113897366,
      Russell2000: 0.0038430890539603462,
      TAIEX: 0.031495131590299374,
    },
    stdSource: '10-seed full rerun (100 epochs)',
  },
  {
    name: 'QENN (paper reported)',
    family: 'Paper',
    meanNmse: {
      BSE: 0.23782,
      NASDAQ: 0.2183,
      HSI: 0.29263,
      SSE: 0.22984,
      Russell2000: 0.25729,
      TAIEX: 0.10163,
    },
  },
  {
    name: 'QENN-Qiskit',
    family: 'Quantum',
    qubits: 5,
    meanNmse: {
      BSE: 0.3595484448113886,
      NASDAQ: 0.3399525988643497,
      HSI: 0.3070773278781621,
      SSE: 0.6114289476283645,
      Russell2000: 0.3489338414167536,
      TAIEX: 0.4413310747090579,
    },
    stdNmse: {
      BSE: 0.15082724471650516,
      NASDAQ: 0.13720745445237095,
      HSI: 0.18297454470097474,
      SSE: 0.3925924145730894,
      Russell2000: 0.1475197008437898,
      TAIEX: 0.20357683696972292,
    },
    stdSource: '10-seed fast approximation (10 epochs)',
  },
  {
    name: 'QENN-QSharp',
    family: 'Quantum',
    qubits: 5,
    meanNmse: {
      BSE: 0.2674826049134368,
      NASDAQ: 0.2403289145521893,
      HSI: 0.2252815214165487,
      SSE: 0.4996617441939071,
      Russell2000: 0.2561938968098615,
      TAIEX: 0.3200449605897392,
    },
    stdNmse: {
      BSE: 0.18247589923020016,
      NASDAQ: 0.15650766357343665,
      HSI: 0.15555449774473623,
      SSE: 0.41351198568659353,
      Russell2000: 0.14838680770918786,
      TAIEX: 0.21025386563779,
    },
    stdSource: '10-seed fast approximation (10 epochs)',
  },
]

export const QENN_GAP_ROWS: GapRow[] = [
  {
    model: 'QENN-Qiskit',
    dataset: 'BSE',
    paperQenn: 0.23782,
    reproduced: 0.3595484448113886,
    absGap: 0.12172844481138859,
    relGapPct: 51.1851168158223,
  },
  {
    model: 'QENN-Qiskit',
    dataset: 'NASDAQ',
    paperQenn: 0.2183,
    reproduced: 0.3399525988643497,
    absGap: 0.1216525988643497,
    relGapPct: 55.727255549404354,
  },
  {
    model: 'QENN-Qiskit',
    dataset: 'HSI',
    paperQenn: 0.29263,
    reproduced: 0.3070773278781621,
    absGap: 0.014447327878162108,
    relGapPct: 4.9370631439572525,
  },
  {
    model: 'QENN-Qiskit',
    dataset: 'SSE',
    paperQenn: 0.22984,
    reproduced: 0.6114289476283645,
    absGap: 0.38158894762836454,
    relGapPct: 166.02373286998107,
  },
  {
    model: 'QENN-Qiskit',
    dataset: 'Russell2000',
    paperQenn: 0.25729,
    reproduced: 0.3489338414167536,
    absGap: 0.09164384141675358,
    relGapPct: 35.618889741829676,
  },
  {
    model: 'QENN-Qiskit',
    dataset: 'TAIEX',
    paperQenn: 0.10163,
    reproduced: 0.4413310747090579,
    absGap: 0.3397010747090579,
    relGapPct: 334.25275480572463,
  },
  {
    model: 'QENN-QSharp',
    dataset: 'BSE',
    paperQenn: 0.23782,
    reproduced: 0.2674826049134368,
    absGap: 0.02966260491343678,
    relGapPct: 12.472712519315776,
  },
  {
    model: 'QENN-QSharp',
    dataset: 'NASDAQ',
    paperQenn: 0.2183,
    reproduced: 0.2403289145521893,
    absGap: 0.022028914552189305,
    relGapPct: 10.091119813187955,
  },
  {
    model: 'QENN-QSharp',
    dataset: 'HSI',
    paperQenn: 0.29263,
    reproduced: 0.2252815214165487,
    absGap: 0.0673484785834513,
    relGapPct: 23.014892042323513,
  },
  {
    model: 'QENN-QSharp',
    dataset: 'SSE',
    paperQenn: 0.22984,
    reproduced: 0.4996617441939071,
    absGap: 0.2698217441939071,
    relGapPct: 117.39546823612388,
  },
  {
    model: 'QENN-QSharp',
    dataset: 'Russell2000',
    paperQenn: 0.25729,
    reproduced: 0.2561938968098615,
    absGap: 0.001096103190138531,
    relGapPct: 0.4260185744251742,
  },
  {
    model: 'QENN-QSharp',
    dataset: 'TAIEX',
    paperQenn: 0.10163,
    reproduced: 0.3200449605897392,
    absGap: 0.2184149605897392,
    relGapPct: 214.91189667395375,
  },
]

export const RUNTIME_ROWS: RuntimeRow[] = [
  {
    model: 'ENN-100',
    epochs: 100,
    totalTrainSec: 2.3442201614379883,
    secPerEpoch: 0.023442201614379883,
    qSharpFallback: 'NA',
  },
  {
    model: 'QENN-Qiskit',
    epochs: 10,
    totalTrainSec: 72.91069436073303,
    secPerEpoch: 7.291069436073303,
    qSharpFallback: 'NA',
  },
  {
    model: 'QENN-QSharp',
    epochs: 10,
    totalTrainSec: 68.87979221343994,
    secPerEpoch: 6.887979221343994,
    qSharpFallback: 'True',
  },
]

export const PROTOCOL_CONTROLS: ProtocolControl[] = [
  {
    parameter: 'Window size',
    value: '6',
    impact: 'Ensures identical temporal context length for all models.',
  },
  {
    parameter: 'Train/test split',
    value: 'Chronological 80/20, no shuffle',
    impact: 'Prevents future leakage from random shuffling in time-series evaluation.',
  },
  {
    parameter: 'Normalization scope',
    value: 'Global min-max to [-1, 1] per dataset',
    impact: 'Keeps value range comparable across models and datasets.',
  },
  {
    parameter: 'Evaluation metric',
    value: 'NMSE (lower is better)',
    impact: 'Single shared objective for fair ranking.',
  },
  {
    parameter: 'Classical variability',
    value: '10-seed full rerun (100 epochs)',
    impact: 'High-confidence ENN standard deviations.',
  },
  {
    parameter: 'Quantum variability',
    value: '10-seed fast approximation',
    impact:
      'QENN std values are from a fast protocol and should be interpreted with that caveat.',
  },
  {
    parameter: 'Q# execution mode',
    value: 'Fallback used in this environment',
    impact:
      'QENN-QSharp values reflect faithful host-side amplitudes rather than active dotnet runtime.',
  },
]

export const MODEL_COLORS: Record<string, string> = {
  'ENN, 10 hidden neurons': '#2563eb',
  'ENN, 20 hidden neurons': '#0ea5e9',
  'ENN, 40 hidden neurons': '#14b8a6',
  'ENN, 70 hidden neurons': '#10b981',
  'ENN, 100 hidden neurons': '#22c55e',
  'QENN (paper reported)': '#f59e0b',
  'QENN-Qiskit': '#f97316',
  'QENN-QSharp': '#8b5cf6',
}
