import React, { createContext, useContext, useState, useEffect } from 'react';

const ViewModeContext = createContext(null);

export const TERMINOLOGY = {
  grain: {
    plain: 'What Each Row Means',
    technical: 'Dataset Grain',
    hint: 'Whether one record is an individual customer, transaction, or summary',
  },
  join: {
    plain: 'Table Link',
    technical: 'SQL Join',
    hint: 'Connecting two tables using matching ID columns',
  },
  joins: {
    plain: 'Table Connections',
    technical: 'Cross-Dataset Joins',
    hint: 'Discovered links between separate tables',
  },
  fanout: {
    plain: 'Double-Counting Warning',
    technical: 'Fan-Out Multiplier (1:N)',
    hint: 'When one record matches multiple records, multiplying totals without grouping',
  },
  measures: {
    plain: 'Totals & Numbers',
    technical: 'Numeric Measures',
    hint: 'Numeric columns (like revenue or score) that can be summed',
  },
  dimensions: {
    plain: 'Categories & Groupings',
    technical: 'Categorical Dimensions',
    hint: 'Attributes used to segment records (like country or plan type)',
  },
  fks: {
    plain: 'Linkable IDs',
    technical: 'Foreign Keys (FKs)',
    hint: 'ID columns used to connect across tables',
  },
  confidence: {
    plain: 'Certainty Score',
    technical: 'Model Confidence',
    hint: 'How certain the platform is about this finding or prediction',
  },
  shap: {
    plain: 'Top Driving Factors',
    technical: 'SHAP Feature Attribution',
    hint: 'The most influential inputs that drove this result',
  },
  lime: {
    plain: 'Cross-Check Verification',
    technical: 'LIME Local Explanation',
    hint: 'Secondary independent check to verify AI explanation consistency',
  },
  conformal: {
    plain: 'Estimated Range',
    technical: 'Conformal Prediction Interval',
    hint: 'The high and low bounds with statistical coverage guarantee',
  },
  drift: {
    plain: 'Accuracy Changes Over Time',
    technical: 'Data & Model Drift',
    hint: 'When real-world data patterns diverge from baseline training',
  },
  ledger: {
    plain: 'Action History & Results',
    technical: 'Decision Ledger',
    hint: 'Log of actions committed, scheduled follow-ups, and business ROI',
  },
  whatif: {
    plain: 'Test Business Scenarios',
    technical: 'What-If Counterfactual Simulation',
    hint: 'Simulate changes to levers before taking actions',
  },
  anomalies: {
    plain: 'Unusual Outliers & Spikes',
    technical: 'Statistical Anomalies (Z > 2.5)',
    hint: 'Records or trends that deviate drastically from normal baselines',
  },
  binary_classification: {
    plain: 'Yes / No Risk Question',
    technical: 'Binary Classification',
    hint: 'Predicting whether an outcome will occur or not (e.g. churn vs stay)',
  },
  regression: {
    plain: 'Predict a Number',
    technical: 'Continuous Regression',
    hint: 'Predicting a numerical amount (e.g. revenue, tenure, spend)',
  },
  target: {
    plain: 'Outcome to Predict',
    technical: 'Target Variable (Label y)',
    hint: 'The column you want the AI model to learn to predict',
  },
  features: {
    plain: 'Contributing Factors',
    technical: 'Input Features (Matrix X)',
    hint: 'The columns used by the model to calculate the outcome',
  },
  positive_label: {
    plain: 'What Counts as a Match',
    technical: 'Target Positive Class (y=1)',
    hint: 'The specific value indicating the outcome of concern (e.g. Yes, Churned)',
  },
  auc: {
    plain: 'Ranking Reliability (AUC)',
    technical: 'ROC-AUC Score',
    hint: 'How well the model separates high-risk cases from low-risk cases (0.5=random, 1.0=perfect)',
  },
  accuracy: {
    plain: 'Overall Match Rate',
    technical: 'Model Accuracy',
    hint: 'Percentage of total predictions that turned out correct',
  },
  precision: {
    plain: 'False Alarm Filter (Precision)',
    technical: 'Precision (PPV)',
    hint: 'When flagged as high priority, how often it actually is true',
  },
  recall: {
    plain: 'Catch Rate (Recall)',
    technical: 'Recall / Sensitivity (TPR)',
    hint: 'Percentage of all real-world occurrences caught by the model',
  },
  r2: {
    plain: 'Pattern Reliability (R²)',
    technical: 'Coefficient of Determination (R²)',
    hint: 'Proportion of variance explained by the model (higher is better)',
  },
  mae: {
    plain: 'Average Estimation Error',
    technical: 'Mean Absolute Error (MAE)',
    hint: 'Average difference between predicted and actual values in original units',
  },
  rmse: {
    plain: 'Root Mean Squared Error',
    technical: 'RMSE Metric',
    hint: 'Penalizes large outlier misses more heavily than standard MAE',
  },
  concordance: {
    plain: 'Agreement Rate with Known History',
    technical: 'Historical Spot-Check Concordance',
    hint: 'How often model ranking matched actual historical outcomes',
  },
  levers: {
    plain: 'Things You Can Change',
    technical: 'Controllable Feature Levers',
    hint: 'Factors within your control that can shift the predicted outcome',
  },
  intervals: {
    plain: 'Expected Prediction Range',
    technical: 'Monte Carlo 10th–90th Percentile Interval',
    hint: 'Realistic low-to-high spread accounting for uncertainty',
  },
  ensemble: {
    plain: 'AI Agreement Level',
    technical: 'Ensemble Consensus',
    hint: 'Agreement score across multiple independent model algorithms (XGBoost, LightGBM, etc.)',
  },
  baseline: {
    plain: 'Starting Benchmark',
    technical: 'Base Value / Prior E[y]',
    hint: 'The average starting prediction before individual factors are weighed',
  },
};

export function ViewModeProvider({ children }) {
  const [plainEnglish, setPlainEnglishState] = useState(() => {
    try {
      const stored = localStorage.getItem('knowa_view_mode');
      if (stored !== null) {
        return stored === 'plain';
      }
    } catch {
      /* ignore */
    }
    return true; // Plain English by default
  });

  const setPlainEnglish = (val) => {
    const isPlain = typeof val === 'function' ? val(plainEnglish) : !!val;
    setPlainEnglishState(isPlain);
    try {
      localStorage.setItem('knowa_view_mode', isPlain ? 'plain' : 'technical');
    } catch {
      /* ignore */
    }
  };

  const toggleMode = () => {
    setPlainEnglish(!plainEnglish);
  };

  const mode = plainEnglish ? 'plain' : 'technical';

  const term = (key, fallback) => {
    const entry = TERMINOLOGY[key];
    if (!entry) return fallback || key;
    return plainEnglish ? entry.plain : entry.technical;
  };

  const hint = (key) => {
    return TERMINOLOGY[key]?.hint || '';
  };

  return (
    <ViewModeContext.Provider
      value={{
        plainEnglish,
        setPlainEnglish,
        toggleMode,
        mode,
        term,
        hint,
      }}
    >
      {children}
    </ViewModeContext.Provider>
  );
}

export function useViewMode() {
  const context = useContext(ViewModeContext);
  if (!context) {
    return {
      plainEnglish: true,
      setPlainEnglish: () => {},
      toggleMode: () => {},
      mode: 'plain',
      term: (k, fb) => fb || k,
      hint: () => '',
    };
  }
  return context;
}
