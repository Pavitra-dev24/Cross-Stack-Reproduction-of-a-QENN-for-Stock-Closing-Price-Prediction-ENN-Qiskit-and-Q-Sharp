import { useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import './App.css'
import {
  DATASETS,
  MODEL_COLORS,
  MODEL_SERIES,
  PROTOCOL_CONTROLS,
  QENN_GAP_ROWS,
  RUNTIME_ROWS,
  type DatasetName,
  type Family,
} from './data/experimentData'

type SummaryMetric = 'nmse' | 'rmse'
type SortKey =
  | 'dataset'
  | 'model'
  | 'family'
  | 'meanNmse'
  | 'rmseEstimate'
  | 'stdNmse'
  | 'cv'
  | 'ci95'
  | 'deltaVsPaper'

const FAMILY_ORDER: Family[] = ['Classical', 'Quantum', 'Paper']
const N_STD_RUNS = 10

function formatMetric(value: number, digits = 5): string {
  return value.toFixed(digits)
}

function formatSignedMetric(value: number, digits = 5): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`
}

function average(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function metricCellColor(value: number): string {
  const minLog = Math.log10(0.003)
  const maxLog = Math.log10(0.8)
  const t = Math.min(
    1,
    Math.max(0, (Math.log10(Math.max(value, 0.000001)) - minLog) / (maxLog - minLog)),
  )

  const low = [22, 163, 74]
  const high = [239, 68, 68]
  const mix = low.map((lo, index) => Math.round(lo + (high[index] - lo) * t))
  return `rgb(${mix[0]},${mix[1]},${mix[2]})`
}

function App() {
  const [selectedDataset, setSelectedDataset] = useState<DatasetName>('BSE')
  const [summaryMetric, setSummaryMetric] = useState<SummaryMetric>('nmse')
  const [familyFilter, setFamilyFilter] = useState<Record<Family, boolean>>({
    Classical: true,
    Quantum: true,
    Paper: true,
  })
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({
    key: 'meanNmse',
    dir: 1,
  })

  const selectedModels = useMemo(
    () => MODEL_SERIES.filter((model) => familyFilter[model.family]),
    [familyFilter],
  )

  const paperBaseline = MODEL_SERIES.find(
    (model) => model.name === 'QENN (paper reported)',
  )!.meanNmse[selectedDataset]

  const meanByModel = selectedModels.map((model) => {
    const nmse = model.meanNmse[selectedDataset]
    return {
      model: model.name,
      family: model.family,
      value: summaryMetric === 'nmse' ? nmse : Math.sqrt(nmse),
      nmse,
    }
  })

  const trendData = DATASETS.map((dataset) => {
    const row: Record<string, string | number> = { dataset }
    selectedModels.forEach((model) => {
      row[model.name] =
        summaryMetric === 'nmse'
          ? model.meanNmse[dataset]
          : Math.sqrt(model.meanNmse[dataset])
    })
    return row
  })

  const stdByModel = selectedModels
    .map((model) => ({
      model: model.name,
      std: model.stdNmse?.[selectedDataset] ?? null,
      source: model.stdSource ?? 'Not available',
    }))
    .filter((row) => row.std !== null) as Array<{
    model: string
    std: number
    source: string
  }>

  const gapByDataset = DATASETS.map((dataset) => ({
    dataset,
    qiskitRelGap:
      QENN_GAP_ROWS.find(
        (row) => row.dataset === dataset && row.model === 'QENN-Qiskit',
      )?.relGapPct ?? 0,
    qsharpRelGap:
      QENN_GAP_ROWS.find(
        (row) => row.dataset === dataset && row.model === 'QENN-QSharp',
      )?.relGapPct ?? 0,
  }))

  const nonPaperModels = selectedModels.filter((model) => model.family !== 'Paper')
  const quantumModels = selectedModels.filter((model) => model.family === 'Quantum')

  const bestModel =
    nonPaperModels.length === 0
      ? null
      : nonPaperModels.reduce((best, current) =>
          current.meanNmse[selectedDataset] < best.meanNmse[selectedDataset]
            ? current
            : best,
        )

  const bestQuantum =
    quantumModels.length === 0
      ? null
      : quantumModels.reduce((best, current) =>
          current.meanNmse[selectedDataset] < best.meanNmse[selectedDataset]
            ? current
            : best,
        )

  const mostStable = stdByModel.length
    ? stdByModel.reduce((best, current) => (current.std < best.std ? current : best))
    : null

  const rankingRows = nonPaperModels
    .map((model) => {
      const meanNmse = model.meanNmse[selectedDataset]
      const std = model.stdNmse?.[selectedDataset] ?? null
      const cv = std === null || Math.abs(meanNmse) < 1e-12 ? null : std / meanNmse
      const ci95 = std === null ? null : (1.96 * std) / Math.sqrt(N_STD_RUNS)
      const deltaToPaper = model.family === 'Quantum' ? meanNmse - paperBaseline : null

      return {
        model: model.name,
        family: model.family,
        meanNmse,
        rmseEstimate: Math.sqrt(meanNmse),
        std,
        cv,
        ci95,
        deltaToPaper,
      }
    })
    .sort((a, b) => a.meanNmse - b.meanNmse)

  const modelSummaryRows = MODEL_SERIES.map((model) => ({
    model: model.name,
    family: model.family,
    capacity:
      model.family === 'Classical'
        ? `${model.hiddenNeurons} hidden neurons`
        : model.family === 'Quantum'
          ? `${model.qubits} qubits (state dim ${2 ** (model.qubits ?? 0)})`
          : 'Reference row',
    avgMeanNmse: average(DATASETS.map((dataset) => model.meanNmse[dataset])),
    avgRmseEstimate: average(DATASETS.map((dataset) => Math.sqrt(model.meanNmse[dataset]))),
    stdSource: model.stdSource ?? 'Not available',
  }))

  const stdMatrixRows = selectedModels.map((model) => ({
    model: model.name,
    family: model.family,
    stdSource: model.stdSource ?? 'Not available',
    values: DATASETS.map((dataset) => model.stdNmse?.[dataset] ?? null),
    avgStd: model.stdNmse
      ? average(DATASETS.map((dataset) => model.stdNmse?.[dataset] ?? 0))
      : null,
  }))

  const radarData = DATASETS.map((dataset) => {
    const row: Record<string, string | number> = { dataset }
    selectedModels.forEach((model) => {
      const nmse = model.meanNmse[dataset]
      row[model.name] = 1 / (1 + nmse)
    })
    return row
  })

  const heatmapRows = selectedModels.map((model) => {
    const values = DATASETS.map((dataset) =>
      summaryMetric === 'nmse'
        ? model.meanNmse[dataset]
        : Math.sqrt(model.meanNmse[dataset]),
    )
    const avg = average(values)
    return { model: model.name, values, avg }
  })

  const fullRows = useMemo(() => {
    const rows: Array<{
      dataset: DatasetName
      model: string
      family: Family
      meanNmse: number
      rmseEstimate: number
      stdNmse: number | null
      cv: number | null
      ci95: number | null
      deltaVsPaper: number | null
      stdSource: string
    }> = []

    DATASETS.forEach((dataset) => {
      const paper = MODEL_SERIES.find((model) => model.name === 'QENN (paper reported)')!
        .meanNmse[dataset]
      selectedModels.forEach((model) => {
        const meanNmse = model.meanNmse[dataset]
        const std = model.stdNmse?.[dataset] ?? null
        const cv = std === null || Math.abs(meanNmse) < 1e-12 ? null : std / meanNmse
        const ci95 = std === null ? null : (1.96 * std) / Math.sqrt(N_STD_RUNS)
        rows.push({
          dataset,
          model: model.name,
          family: model.family,
          meanNmse,
          rmseEstimate: Math.sqrt(meanNmse),
          stdNmse: std,
          cv,
          ci95,
          deltaVsPaper: model.family === 'Quantum' ? meanNmse - paper : null,
          stdSource: model.stdSource ?? 'Not available',
        })
      })
    })

    return rows
  }, [selectedModels])

  const sortedRows = [...fullRows].sort((a, b) => {
    const { key, dir } = sort
    const av = a[key]
    const bv = b[key]
    if (typeof av === 'string' && typeof bv === 'string') {
      return av.localeCompare(bv) * dir
    }
    if (typeof av === 'number' && typeof bv === 'number') {
      return (av - bv) * dir
    }
    if (av === null && bv !== null) return 1
    if (av !== null && bv === null) return -1
    return 0
  })

  const toggleFamily = (family: Family): void => {
    setFamilyFilter((current) => ({
      ...current,
      [family]: !current[family],
    }))
  }

  const onSort = (key: SortKey): void => {
    setSort((current) => ({
      key,
      dir: current.key === key ? (current.dir === 1 ? -1 : 1) : 1,
    }))
  }

  return (
    <div className="dashboard">
      <header className="page-header">
        <h1>Stock Prediction Experiment Dashboard</h1>
        <p>
          Fully based on our own research pipeline outputs. No external repo result
          values are used.
        </p>
      </header>

      <section className="controls card">
        <div className="control-group">
          <label htmlFor="dataset">Dataset</label>
          <select
            id="dataset"
            value={selectedDataset}
            onChange={(event) => setSelectedDataset(event.target.value as DatasetName)}
          >
            {DATASETS.map((dataset) => (
              <option key={dataset} value={dataset}>
                {dataset}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <span>Model families</span>
          <div className="checkbox-row">
            {FAMILY_ORDER.map((family) => (
              <label key={family}>
                <input
                  type="checkbox"
                  checked={familyFilter[family]}
                  onChange={() => toggleFamily(family)}
                />
                {family}
              </label>
            ))}
          </div>
        </div>

        <div className="control-group">
          <span>Primary metric</span>
          <div className="metric-toggle-row">
            <button
              type="button"
              className={summaryMetric === 'nmse' ? 'selected' : ''}
              onClick={() => setSummaryMetric('nmse')}
            >
              NMSE
            </button>
            <button
              type="button"
              className={summaryMetric === 'rmse' ? 'selected' : ''}
              onClick={() => setSummaryMetric('rmse')}
            >
              RMSE (derived)
            </button>
          </div>
        </div>
      </section>

      <section className="metrics-grid">
        <MetricCard
          title="Best model on selected dataset"
          value={
            bestModel
              ? `${bestModel.name} (${formatMetric(bestModel.meanNmse[selectedDataset])})`
              : 'N/A'
          }
          subtitle="Lowest mean NMSE"
        />
        <MetricCard
          title="Best quantum model"
          value={
            bestQuantum
              ? `${bestQuantum.name} (${formatMetric(bestQuantum.meanNmse[selectedDataset])})`
              : 'N/A'
          }
          subtitle="Lowest mean NMSE within quantum models"
        />
        <MetricCard
          title="Paper QENN baseline"
          value={formatMetric(paperBaseline)}
          subtitle={`Reference for ${selectedDataset}`}
        />
        <MetricCard
          title="Most stable (lowest std)"
          value={mostStable ? `${mostStable.model} (${formatMetric(mostStable.std)})` : 'N/A'}
          subtitle="From available std rows"
        />
      </section>

      <section className="charts-grid">
        <div className="card chart-card">
          <h2>
            {summaryMetric === 'nmse' ? 'Mean NMSE' : 'Derived RMSE'} by model (
            {selectedDataset})
          </h2>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart
              data={meanByModel}
              margin={{ top: 12, right: 16, left: 8, bottom: 80 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="model"
                angle={-32}
                textAnchor="end"
                interval={0}
                height={88}
              />
              <YAxis />
              <Tooltip
                formatter={(value) => formatMetric(Number(value))}
                labelFormatter={(label) => `Model: ${label}`}
              />
              <Legend />
              {summaryMetric === 'nmse' && (
                <ReferenceLine
                  y={paperBaseline}
                  stroke="#f59e0b"
                  strokeDasharray="4 4"
                  label="Paper QENN"
                />
              )}
              <Bar
                dataKey="value"
                name={summaryMetric === 'nmse' ? 'Mean NMSE' : 'Derived RMSE'}
              >
                {meanByModel.map((row) => (
                  <Cell key={row.model} fill={MODEL_COLORS[row.model] ?? '#6366f1'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card chart-card">
          <h2>
            {summaryMetric === 'nmse' ? 'Mean NMSE' : 'Derived RMSE'} trend across
            datasets
          </h2>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={trendData} margin={{ top: 12, right: 16, left: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="dataset" />
              <YAxis />
              <Tooltip formatter={(value) => formatMetric(Number(value))} />
              <Legend />
              {selectedModels.map((model) => (
                <Line
                  key={model.name}
                  type="monotone"
                  dataKey={model.name}
                  stroke={MODEL_COLORS[model.name] ?? '#6366f1'}
                  strokeWidth={2}
                  dot={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="card chart-card">
          <h2>Standard deviation ({selectedDataset})</h2>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart
              data={stdByModel}
              margin={{ top: 12, right: 16, left: 8, bottom: 80 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="model"
                angle={-32}
                textAnchor="end"
                interval={0}
                height={88}
              />
              <YAxis />
              <Tooltip
                formatter={(value) => formatMetric(Number(value))}
                labelFormatter={(label) => `Model: ${label}`}
              />
              <Legend />
              <Bar dataKey="std" name="Std dev NMSE">
                {stdByModel.map((row) => (
                  <Cell key={row.model} fill={MODEL_COLORS[row.model] ?? '#6366f1'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card chart-card">
          <h2>Quantum relative gap vs paper QENN</h2>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={gapByDataset} margin={{ top: 12, right: 16, left: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="dataset" />
              <YAxis />
              <Tooltip formatter={(value) => `${Number(value).toFixed(2)}%`} />
              <Legend />
              <Bar dataKey="qiskitRelGap" name="QENN-Qiskit rel gap %" fill="#f97316" />
              <Bar dataKey="qsharpRelGap" name="QENN-QSharp rel gap %" fill="#0ea5e9" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card chart-card">
          <h2>Radar comparison (score = 1 / (1 + NMSE))</h2>
          <ResponsiveContainer width="100%" height={320}>
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="dataset" />
              <Tooltip
                formatter={(value) => {
                  const score = Number(value)
                  const nmse = score > 0 ? 1 / score - 1 : Infinity
                  return Number.isFinite(nmse) ? nmse.toFixed(5) : 'N/A'
                }}
              />
              <Legend />
              {selectedModels.map((model) => (
                <Radar
                  key={model.name}
                  dataKey={model.name}
                  stroke={MODEL_COLORS[model.name] ?? '#6366f1'}
                  fill={MODEL_COLORS[model.name] ?? '#6366f1'}
                  fillOpacity={0.12}
                />
              ))}
            </RadarChart>
          </ResponsiveContainer>
        </div>

        <div className="card chart-card">
          <h2>Runtime profile (representative BSE runs)</h2>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart
              data={RUNTIME_ROWS}
              margin={{ top: 12, right: 16, left: 8, bottom: 40 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="model" />
              <YAxis />
              <Tooltip formatter={(value) => Number(value).toFixed(3)} />
              <Legend />
              <Bar dataKey="secPerEpoch" name="Seconds per epoch" fill="#8b5cf6" />
            </BarChart>
          </ResponsiveContainer>
          <p className="subtle">
            Runtime rows are representative only (epochs differ across model profiles).
          </p>
        </div>
      </section>

      <section className="card">
        <h2>
          {summaryMetric === 'nmse' ? 'NMSE' : 'Derived RMSE'} heatmap (models ×
          datasets)
        </h2>
        <div className="heatmap-wrap">
          <table className="heatmap-table">
            <thead>
              <tr>
                <th>Model</th>
                {DATASETS.map((dataset) => (
                  <th key={dataset}>{dataset}</th>
                ))}
                <th>Avg</th>
              </tr>
            </thead>
            <tbody>
              {heatmapRows.map((row) => (
                <tr key={row.model}>
                  <td>{row.model}</td>
                  {row.values.map((value, idx) => (
                    <td key={`${row.model}-${idx}`}>
                      <span
                        className="heat-cell"
                        style={{ backgroundColor: metricCellColor(value) }}
                      >
                        {value.toFixed(3)}
                      </span>
                    </td>
                  ))}
                  <td>
                    <span
                      className="heat-cell"
                      style={{ backgroundColor: metricCellColor(row.avg) }}
                    >
                      {row.avg.toFixed(3)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="tables-grid">
        <div className="card">
          <h2>Model setup and comparability context</h2>
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Family</th>
                <th>Capacity</th>
                <th>Avg mean NMSE</th>
                <th>Avg RMSE (derived)</th>
                <th>Std source</th>
              </tr>
            </thead>
            <tbody>
              {modelSummaryRows.map((row) => (
                <tr key={row.model}>
                  <td>{row.model}</td>
                  <td>{row.family}</td>
                  <td>{row.capacity}</td>
                  <td>{formatMetric(row.avgMeanNmse)}</td>
                  <td>{formatMetric(row.avgRmseEstimate)}</td>
                  <td>{row.stdSource}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="card">
          <h2>Selected dataset ranking ({selectedDataset})</h2>
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Family</th>
                <th>Mean NMSE</th>
                <th>RMSE (derived)</th>
                <th>Std dev</th>
                <th>CV</th>
                <th>95% CI half-width</th>
                <th>Delta vs paper QENN</th>
              </tr>
            </thead>
            <tbody>
              {rankingRows.map((row) => (
                <tr key={row.model}>
                  <td>{row.model}</td>
                  <td>{row.family}</td>
                  <td>{formatMetric(row.meanNmse)}</td>
                  <td>{formatMetric(row.rmseEstimate)}</td>
                  <td>{row.std === null ? 'N/A' : formatMetric(row.std)}</td>
                  <td>{row.cv === null ? 'N/A' : formatMetric(row.cv)}</td>
                  <td>{row.ci95 === null ? 'N/A' : formatMetric(row.ci95)}</td>
                  <td>
                    {row.deltaToPaper === null
                      ? 'N/A'
                      : formatSignedMetric(row.deltaToPaper)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card">
        <h2>Standard deviation table (NMSE, local values)</h2>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Family</th>
                {DATASETS.map((dataset) => (
                  <th key={dataset}>{dataset}</th>
                ))}
                <th>Avg std</th>
                <th>Source</th>
              </tr>
            </thead>
            <tbody>
              {stdMatrixRows.map((row) => (
                <tr key={row.model}>
                  <td>{row.model}</td>
                  <td>{row.family}</td>
                  {row.values.map((value, idx) => (
                    <td key={`${row.model}-std-${idx}`}>
                      {value === null ? 'N/A' : formatMetric(value)}
                    </td>
                  ))}
                  <td>{row.avgStd === null ? 'N/A' : formatMetric(row.avgStd)}</td>
                  <td>{row.stdSource}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card">
        <h2>Full sortable comparison table (all datasets × selected models)</h2>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <SortableHeader
                  label="Dataset"
                  keyName="dataset"
                  active={sort}
                  onSort={onSort}
                />
                <SortableHeader label="Model" keyName="model" active={sort} onSort={onSort} />
                <SortableHeader
                  label="Family"
                  keyName="family"
                  active={sort}
                  onSort={onSort}
                />
                <SortableHeader
                  label="Mean NMSE"
                  keyName="meanNmse"
                  active={sort}
                  onSort={onSort}
                />
                <SortableHeader
                  label="RMSE (derived)"
                  keyName="rmseEstimate"
                  active={sort}
                  onSort={onSort}
                />
                <SortableHeader
                  label="Std NMSE"
                  keyName="stdNmse"
                  active={sort}
                  onSort={onSort}
                />
                <SortableHeader label="CV" keyName="cv" active={sort} onSort={onSort} />
                <SortableHeader
                  label="95% CI"
                  keyName="ci95"
                  active={sort}
                  onSort={onSort}
                />
                <SortableHeader
                  label="Delta vs paper"
                  keyName="deltaVsPaper"
                  active={sort}
                  onSort={onSort}
                />
              </tr>
            </thead>
            <tbody>
              {sortedRows.map((row, index) => (
                <tr key={`${row.dataset}-${row.model}-${index}`}>
                  <td>{row.dataset}</td>
                  <td>{row.model}</td>
                  <td>{row.family}</td>
                  <td>{formatMetric(row.meanNmse)}</td>
                  <td>{formatMetric(row.rmseEstimate)}</td>
                  <td>{row.stdNmse === null ? 'N/A' : formatMetric(row.stdNmse)}</td>
                  <td>{row.cv === null ? 'N/A' : formatMetric(row.cv)}</td>
                  <td>{row.ci95 === null ? 'N/A' : formatMetric(row.ci95)}</td>
                  <td>
                    {row.deltaVsPaper === null
                      ? 'N/A'
                      : formatSignedMetric(row.deltaVsPaper)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card">
        <h2>Fair-comparison parameters and protocol controls</h2>
        <table>
          <thead>
            <tr>
              <th>Parameter</th>
              <th>Value</th>
              <th>Why it matters</th>
            </tr>
          </thead>
          <tbody>
            {PROTOCOL_CONTROLS.map((control) => (
              <tr key={control.parameter}>
                <td>{control.parameter}</td>
                <td>{control.value}</td>
                <td>{control.impact}</td>
              </tr>
            ))}
            <tr>
              <td>Derived RMSE</td>
              <td>sqrt(NMSE)</td>
              <td>
                Derived from existing NMSE means on normalized scale (no external values).
              </td>
            </tr>
            <tr>
              <td>95% confidence interval</td>
              <td>1.96 * std / sqrt(10)</td>
              <td>Computed where std is available, using run count n = 10.</td>
            </tr>
            <tr>
              <td>Excluded diagnostics</td>
              <td>pred/actual, loss curves, MAPE timelines</td>
              <td>
                Removed because they require per-sample/per-epoch traces not available in the
                current local aggregates.
              </td>
            </tr>
          </tbody>
        </table>
      </section>
    </div>
  )
}

type MetricCardProps = {
  title: string
  value: string
  subtitle: string
}

function MetricCard({ title, value, subtitle }: MetricCardProps) {
  return (
    <article className="card metric-card">
      <h2>{title}</h2>
      <p className="metric-value">{value}</p>
      <p className="subtle">{subtitle}</p>
    </article>
  )
}

type SortableHeaderProps = {
  label: string
  keyName: SortKey
  active: { key: SortKey; dir: 1 | -1 }
  onSort: (key: SortKey) => void
}

function SortableHeader({ label, keyName, active, onSort }: SortableHeaderProps) {
  const marker = active.key === keyName ? (active.dir === 1 ? ' ↑' : ' ↓') : ''
  return (
    <th
      role="button"
      tabIndex={0}
      onClick={() => onSort(keyName)}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault()
          onSort(keyName)
        }
      }}
      className="sortable-header"
    >
      {label}
      {marker}
    </th>
  )
}

export default App
