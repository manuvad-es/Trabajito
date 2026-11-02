using System;
using NavigationDJIA.Interfaces;
using NavigationDJIA.World;
using QMind;
using QMind.Interfaces;
using UnityEngine;

namespace GrupoA
{
    public class QMindTrainer : IQMindTrainer
    {
        private const float CaptureReward = -100f;
        private const float MinEpsilon = 0.05f;

        private QMindTrainerParams _params;
        private WorldInfo _worldInfo;
        INavigationAlgorithm _navigationAlgorithm;

        private QTableStorage _qStorage;
        private QTable _qTable;

        private CellInfo _agentPosition;
        private CellInfo _otherPosition;

        private float _return;
        private float _returnAveraged;
        private System.Random _random = new System.Random();

        private bool _finished;

        #region IQMindTrainer implementation

        public CellInfo AgentPosition => _agentPosition;
        public CellInfo OtherPosition => _otherPosition;

        public int CurrentEpisode { get; private set; }
        public int CurrentStep { get; private set; }

        public float Return => _return;
        public float ReturnAveraged => _returnAveraged;

        public event EventHandler OnEpisodeStarted;
        public event EventHandler OnEpisodeFinished;

        #endregion

        public void Initialize(QMindTrainerParams qMindTrainerParams, WorldInfo worldInfo, INavigationAlgorithm navigationAlgorithm)
        {
            _params = qMindTrainerParams;
            _worldInfo = worldInfo;
            _navigationAlgorithm = navigationAlgorithm;
            _navigationAlgorithm.Initialize(worldInfo);

            // Si ya existe TablaQ.csv se carga y el entrenamiento continúa sobre ella
            _qStorage = new QTableStorage("TablaQ.csv");
            _qTable = new QTable(_qStorage);

            CurrentEpisode = 0;
            _finished = false;
            StartNewEpisode();
        }

        private void StartNewEpisode()
        {
            CurrentEpisode++;
            CurrentStep = 0;
            _return = 0f;
            _returnAveraged = 0f;

            _agentPosition = RandomWalkableCell();
            _otherPosition = RandomWalkableCell();

            // Evita empezar el episodio ya capturado
            int guard = 0;
            while (SameCell(_agentPosition, _otherPosition) && guard++ < 100)
            {
                _otherPosition = RandomWalkableCell();
            }

            OnEpisodeStarted?.Invoke(this, EventArgs.Empty);
        }

        private void EndEpisode()
        {
            bool lastEpisode = _params.episodes > 0 && CurrentEpisode >= _params.episodes;
            int saveEvery = Mathf.Max(1, _params.episodesBetweenSaves);

            if (lastEpisode || CurrentEpisode % saveEvery == 0)
            {
                _qTable.SaveToCsv();
                Debug.Log($"[GrupoA] Episodio {CurrentEpisode}: {CurrentStep} pasos. Tabla Q guardada ({_qStorage.Data.Count} estados).");
            }

            OnEpisodeFinished?.Invoke(this, EventArgs.Empty);

            if (lastEpisode)
            {
                _finished = true;
                return;
            }

            StartNewEpisode();
        }

        public void DoStep(bool train)
        {
            if (_finished) return;

            // Estado actual del agente
            string stateKey = BuildStateKey(_agentPosition, _otherPosition, _worldInfo);

            // Seleciona la acción a realizar
            QAction action = ChooseAction(stateKey, train);

            // Nuevos estados del agente y del oponente
            CellInfo newAgentPos = ApplyAction(_agentPosition, action);
            CellInfo newOtherPos = MoveOpponent(_otherPosition, newAgentPos);

            // Nuevo estado del agente
            string nextStateKey = BuildStateKey(newAgentPos, newOtherPos, _worldInfo);

            // Calcula la recompensa y si el nuevo estado es terminal
            float reward = ComputeReward(newAgentPos, newOtherPos);
            bool terminal = IsTerminalState(newAgentPos, newOtherPos);

            if (train)
            {
                UpdateQ(stateKey, action, reward, nextStateKey, terminal);
            }

            // actualiza las posiciones
            _agentPosition = newAgentPos;
            _otherPosition = newOtherPos;

            // Actualizamos estadísticas de recompensas
            CurrentStep++;
            _return += reward;
            _returnAveraged = (_returnAveraged * (CurrentStep - 1) + reward) / CurrentStep;

            // Fin de episodio: captura o límite de pasos (maxSteps = -1 -> sin límite)
            bool maxStepsReached = _params.maxSteps > 0 && CurrentStep >= _params.maxSteps;
            if (terminal || maxStepsReached)
            {
                EndEpisode();
            }
        }

        #region Parte a implementar por el alumno

        private string BuildStateKey(CellInfo agent, CellInfo other, WorldInfo _worldInfo)
        {
            var state = new QState(agent, other, _worldInfo);
            return state.ToKey();
        }

        /// <summary>
        /// Política epsilon-greedy. Si hay número de episodios definido, epsilon decae
        /// linealmente desde _params.epsilon hasta MinEpsilon a lo largo del entrenamiento.
        /// </summary>
        private QAction ChooseAction(string stateKey, bool train)
        {
            if (!train)
                return _qTable.GetBestAction(stateKey);

            if (_random.NextDouble() < CurrentEpsilon())
            {
                int n = Enum.GetValues(typeof(QAction)).Length;
                return (QAction)_random.Next(n);
            }

            return _qTable.GetBestAction(stateKey);
        }

        private float CurrentEpsilon()
        {
            if (_params.episodes <= 0)
                return _params.epsilon;

            float progress = (float)(CurrentEpisode - 1) / _params.episodes;
            return Mathf.Max(MinEpsilon, _params.epsilon * (1f - progress));
        }

        /// <summary>
        /// Q(s,a) = (1 - alpha) * Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a')).
        /// Si s' es terminal no hay retorno futuro: el objetivo es sólo la recompensa.
        /// </summary>
        private void UpdateQ(string stateKey, QAction action, float reward, string nextStateKey, bool terminal)
        {
            float oldQ = _qTable.GetQ(stateKey, action);
            float maxQNext = terminal ? 0f : _qTable.GetMaxQ(nextStateKey);
            float target = reward + _params.gamma * maxQNext;
            float newQ = (1f - _params.alpha) * oldQ + _params.alpha * target;
            _qTable.SetQ(stateKey, action, newQ);
        }

        /// <summary>
        /// Recompensa:
        ///   - Captura: -100.
        ///   - Sobrevivir un paso: +1, más un pequeño extra proporcional a la distancia
        ///     Manhattan al enemigo (0.1 por casilla, saturado en 10) para premiar alejarse.
        /// </summary>
        private float ComputeReward(CellInfo agent, CellInfo other)
        {
            if (SameCell(agent, other))
                return CaptureReward;

            int distance = Math.Abs(agent.x - other.x) + Math.Abs(agent.y - other.y);
            return 2f + 0.1f * Math.Min(distance, 10);
        }

        /// <summary>Estado terminal: el enemigo alcanza la casilla del agente.</summary>
        private bool IsTerminalState(CellInfo agent, CellInfo other)
        {
            return SameCell(agent, other);
        }

        /// <summary>
        /// Aplica la acción. Si el destino es un muro o está fuera del tablero,
        /// el agente se queda en su casilla.
        /// </summary>
        private CellInfo ApplyAction(CellInfo agentCell, QAction action)
        {
            int nx = agentCell.x;
            int ny = agentCell.y;

            switch (action)
            {
                case QAction.Up:    ny += 1; break;
                case QAction.Down:  ny -= 1; break;
                case QAction.Right: nx += 1; break;
                case QAction.Left:  nx -= 1; break;
                case QAction.Stay:  return agentCell;
            }

            if (QState.IsBlocked(_worldInfo, nx, ny))
                return agentCell;

            return _worldInfo[nx, ny];
        }

        private CellInfo MoveOpponent(CellInfo opponent, CellInfo target)
        {
            var path = _navigationAlgorithm.GetPath(opponent, target, 1);
            if (path != null && path.Length > 0)
                return path[0];

            return opponent;
        }

        private CellInfo RandomWalkableCell()
        {
            CellInfo cell = _worldInfo.RandomCell();
            int guard = 0;
            while (!cell.Walkable && guard++ < 1000)
            {
                cell = _worldInfo.RandomCell();
            }
            return cell;
        }

        private static bool SameCell(CellInfo a, CellInfo b)
        {
            return a.x == b.x && a.y == b.y;
        }

        #endregion
    }
}
