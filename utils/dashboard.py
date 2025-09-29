"""
Clean SOAP Evaluation Dashboard
================================
Simplified, focused dashboard with essential metrics only.
"""

import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import os
import logging

logger = logging.getLogger(__name__)


class SOAPEvaluationDashboard:
    """Clean, simple dashboard for SOAP evaluation metrics."""

    COLORS = {
        'primary': '#2563EB',
        'success': '#10B981',
        'warning': '#F59E0B',
        'danger': '#EF4444',
        'secondary': '#6B7280',
        'dark': '#1F2937'
    }

    def __init__(self, results_files: List[str], dashboard_title: str = "SOAP Evaluation Dashboard"):
        self.results_files = results_files
        self.dashboard_title = dashboard_title
        self.data = self._load_and_process_data()

    def _load_and_process_data(self) -> pd.DataFrame:
        """Load and process all results files."""
        all_data = []

        for file_path in self.results_files:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue

            try:
                with open(file_path, 'r') as f:
                    content = f.read().strip()

                if content.startswith('['):
                    records = json.loads(content)
                else:
                    records = [json.loads(line) for line in content.split(
                        '\n') if line.strip()]

                for record in records:
                    if 'evaluation_metrics' in record:
                        row = self._extract_metrics(record, file_path)
                        if row:
                            all_data.append(row)

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

        if not all_data:
            logger.warning("No valid data found")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.sort_values('timestamp')

        return df

    def _extract_metrics(self, record: Dict[str, Any], source_file: str) -> Optional[Dict[str, Any]]:
        """Extract metrics from a record."""
        try:
            metrics = record['evaluation_metrics']
            det_metrics = metrics.get('deterministic_metrics', {})
            llm_metrics = metrics.get('llm_metrics', {})

            content_fidelity = llm_metrics.get('content_fidelity', {})
            medical_correctness = llm_metrics.get('medical_correctness', {})

            row = {
                'source_file': os.path.basename(source_file),
                'timestamp': record.get('timestamp', datetime.now().isoformat()),
                'engine_type': record.get('engine_type', 'unknown'),

                'entity_coverage': det_metrics.get('entity_coverage', 0),
                'section_completeness': det_metrics.get('section_completeness', 0),
                'format_validity': det_metrics.get('format_validity', 0),

                'content_fidelity_f1': content_fidelity.get('f1', 0) * 100,
                'content_fidelity_precision': content_fidelity.get('precision', 0) * 100,
                'content_fidelity_recall': content_fidelity.get('recall', 0) * 100,
                'medical_accuracy': medical_correctness.get('accuracy', 0) * 100,

                'correctly_captured': content_fidelity.get('counts', {}).get('correctly_captured', 0),
                'missed_critical': content_fidelity.get('counts', {}).get('missed_critical', 0),
                'unsupported_content': content_fidelity.get('counts', {}).get('unsupported_content', 0),

                'overall_quality': self._calculate_overall_quality(det_metrics, llm_metrics),
                'has_errors': 'error' in record
            }
            return row

        except Exception as e:
            logger.error(f"Error extracting metrics: {e}")
            return None

    def _calculate_overall_quality(self, det_metrics: Dict, llm_metrics: Dict) -> float:
        """Calculate weighted overall quality score."""
        try:
            content_f1 = llm_metrics.get('content_fidelity', {}).get('f1', 0)
            medical_acc = llm_metrics.get(
                'medical_correctness', {}).get('accuracy', 0)
            entity_cov = det_metrics.get('entity_coverage', 0) / 100
            section_comp = det_metrics.get('section_completeness', 0) / 100

            quality = (content_f1 * 0.4 + medical_acc * 0.35 +
                       entity_cov * 0.15 + section_comp * 0.1) * 100
            return min(100, max(0, quality))
        except:
            return 0

    def create_comprehensive_dashboard(self, output_file: str = "results/dashboard.html") -> str:
        """Create clean, focused dashboard."""
        if self.data.empty:
            logger.error("No data available")
            return ""

        # Create 2x3 layout for better organization
        fig = make_subplots(
            rows=2, cols=3,
            row_heights=[0.45, 0.55],
            column_widths=[0.4, 0.3, 0.3],
            subplot_titles=[
                'Quality Metrics Timeline',
                'Score Distribution',
                'Summary Statistics',
                'Content Fidelity Breakdown',
                'Metric Comparison',
                'Issues Count'
            ],
            specs=[
                [{"secondary_y": False}, {"type": "box"}, {"type": "table"}],
                [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )

        # 1. Quality Timeline - Simplified with just Overall Quality
        if 'timestamp' in self.data.columns and len(self.data) > 1:
            # Main line - Overall Quality
            fig.add_trace(
                go.Scatter(
                    x=self.data['timestamp'],
                    y=self.data['overall_quality'],
                    mode='lines+markers',
                    name='Overall Quality',
                    line=dict(color=self.COLORS['primary'], width=3.5),
                    marker=dict(size=10, symbol='circle',
                                line=dict(width=2, color='white')),
                    hovertemplate='<b>Overall Quality</b><br>%{y:.1f}%<br>%{x}<extra></extra>'
                ),
                row=1, col=1
            )

            # Add subtle moving average if enough data
            if len(self.data) >= 5:
                ma_window = min(5, len(self.data) // 2)
                ma = self.data['overall_quality'].rolling(
                    window=ma_window, center=True).mean()
                fig.add_trace(
                    go.Scatter(
                        x=self.data['timestamp'],
                        y=ma,
                        mode='lines',
                        name='Trend',
                        line=dict(
                            color=self.COLORS['danger'], width=2, dash='dash'),
                        hovertemplate='<b>Trend</b><br>%{y:.1f}%<extra></extra>'
                    ),
                    row=1, col=1
                )
        else:
            # If no timestamp, use index
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(self.data))),
                    y=self.data['overall_quality'],
                    mode='lines+markers',
                    name='Overall Quality',
                    line=dict(color=self.COLORS['primary'], width=3.5),
                    marker=dict(size=10, line=dict(width=2, color='white'))
                ),
                row=1, col=1
            )

        # 2. Score Distribution Box Plot
        fig.add_trace(
            go.Box(
                y=self.data['overall_quality'],
                name='Quality',
                marker=dict(color=self.COLORS['primary']),
                boxmean='sd',
                showlegend=False,
                hovertemplate='<b>Quality Score</b><br>%{y:.1f}%<extra></extra>'
            ),
            row=1, col=2
        )

        # 3. Summary Statistics Table
        avg_quality = self.data['overall_quality'].mean()
        success_rate = (1 - self.data['has_errors'].mean()) * 100

        stats_color = (self.COLORS['success'] if avg_quality >= 85 else
                       self.COLORS['warning'] if avg_quality >= 70 else
                       self.COLORS['danger'])

        kpi_data = {
            'Metric': [
                'Total Samples',
                'Avg Quality Score',
                'Best Score',
                'Worst Score',
                'Content Fidelity',
                'Medical Accuracy',
                'Success Rate'
            ],
            'Value': [
                str(len(self.data)),
                f"{avg_quality:.1f}%",
                f"{self.data['overall_quality'].max():.1f}%",
                f"{self.data['overall_quality'].min():.1f}%",
                f"{self.data['content_fidelity_f1'].mean():.1f}%",
                f"{self.data['medical_accuracy'].mean():.1f}%",
                f"{success_rate:.1f}%"
            ]
        }

        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>Metric</b>', '<b>Value</b>'],
                    fill_color=self.COLORS['dark'],
                    font=dict(color='white', size=12),
                    align='left',
                    height=32
                ),
                cells=dict(
                    values=[kpi_data['Metric'], kpi_data['Value']],
                    fill_color=['#F9FAFB', 'white'],
                    font=dict(size=11),
                    align='left',
                    height=28
                )
            ),
            row=1, col=3
        )

        # 4. Content Fidelity Breakdown - with proper spacing
        fidelity_totals = {
            'Correctly Captured': self.data['correctly_captured'].sum(),
            'Missed Critical': self.data['missed_critical'].sum(),
            'Unsupported': self.data['unsupported_content'].sum()
        }

        fig.add_trace(
            go.Bar(
                x=list(fidelity_totals.keys()),
                y=list(fidelity_totals.values()),
                marker_color=[self.COLORS['success'],
                              self.COLORS['danger'], self.COLORS['warning']],
                text=list(fidelity_totals.values()),
                textposition='outside',
                textfont=dict(size=14, color=self.COLORS['dark']),
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
            ),
            row=2, col=1
        )

        # 5. Metric Comparison - with better labels
        metrics = {
            'Content Fidelity': self.data['content_fidelity_f1'].mean(),
            'Medical Accuracy': self.data['medical_accuracy'].mean(),
            'Entity Coverage': self.data['entity_coverage'].mean(),
            'Section Complete': self.data['section_completeness'].mean()
        }

        colors = [self.COLORS['success'] if v >= 85 else
                  self.COLORS['warning'] if v >= 70 else
                  self.COLORS['danger'] for v in metrics.values()]

        fig.add_trace(
            go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                marker_color=colors,
                text=[f"{v:.1f}%" for v in metrics.values()],
                textposition='outside',
                textfont=dict(size=14, color=self.COLORS['dark']),
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>Score: %{y:.1f}%<extra></extra>'
            ),
            row=2, col=2
        )

        # 6. Issues Count - with better labels
        issues = {
            'Missed Critical': self.data['missed_critical'].sum(),
            'Unsupported': self.data['unsupported_content'].sum(),
            'Errors': int(self.data['has_errors'].sum())
        }

        fig.add_trace(
            go.Bar(
                x=list(issues.keys()),
                y=list(issues.values()),
                marker_color=[self.COLORS['danger'],
                              self.COLORS['warning'], self.COLORS['secondary']],
                text=list(issues.values()),
                textposition='outside',
                textfont=dict(size=14, color=self.COLORS['dark']),
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
            ),
            row=2, col=3
        )

        # Layout
        fig.update_layout(
            title={
                'text': f"<b>{self.dashboard_title}</b><br><sub style='color:#6B7280'>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}</sub>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 26, 'color': self.COLORS['dark']}
            },
            height=1100,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.05,
                xanchor="center",
                x=0.2,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#E5E7EB",
                borderwidth=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='#F9FAFB',
            font=dict(family="Inter, -apple-system, system-ui, sans-serif",
                      size=11, color=self.COLORS['dark']),
            margin=dict(l=70, r=70, t=120, b=80)
        )

        # Update all axes
        fig.update_xaxes(showgrid=False, zeroline=False, showline=True,
                         linewidth=1, linecolor='#E5E7EB', tickangle=0)
        fig.update_yaxes(showgrid=True, gridcolor='#F3F4F6', gridwidth=1,
                         zeroline=False, showline=True, linewidth=1, linecolor='#E5E7EB')

        # Set y-axis ranges for percentage charts
        fig.update_yaxes(range=[0, 105], row=1, col=1)
        fig.update_yaxes(range=[0, 105], row=1, col=2)
        fig.update_yaxes(range=[0, 105], row=2, col=2)

        # Add extra space for text labels on bar charts
        max_fidelity = max(fidelity_totals.values())
        fig.update_yaxes(range=[0, max_fidelity * 1.15], row=2, col=1)

        max_issues = max(issues.values())
        fig.update_yaxes(range=[0, max_issues * 1.15], row=2, col=3)

        # Axis titles
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Score (%)", row=1, col=1)
        fig.update_yaxes(title_text="Score (%)", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="Score (%)", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=3)

        # Save
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        fig.write_html(
            output_file,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d']
            }
        )
        logger.info(f"Dashboard saved to {output_file}")

        return output_file

    def create_quality_report(self, output_file: str = "results/quality_report.html") -> str:
        """Create detailed quality report."""
        if self.data.empty:
            return ""

        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.25, 0.40, 0.35],
            subplot_titles=[
                'Performance Summary by Source',
                'Detailed Metrics Comparison',
                'Issue Analysis'
            ],
            specs=[
                [{"type": "table"}],
                [{"type": "bar"}],
                [{"type": "bar"}]
            ],
            vertical_spacing=0.12
        )

        # 1. Performance by Source File
        if len(self.data['source_file'].unique()) > 0:
            file_stats = self.data.groupby('source_file').agg({
                'overall_quality': ['mean', 'count'],
                'content_fidelity_f1': 'mean',
                'medical_accuracy': 'mean',
                'has_errors': 'sum'
            }).round(1)

            fig.add_trace(
                go.Table(
                    header=dict(
                        values=['<b>Source File</b>', '<b>Avg Quality</b>', '<b>Samples</b>',
                                '<b>Content F1</b>', '<b>Medical Acc</b>', '<b>Errors</b>'],
                        fill_color=self.COLORS['dark'],
                        font=dict(color='white', size=12),
                        align='left',
                        height=35
                    ),
                    cells=dict(
                        values=[
                            file_stats.index,
                            [f"{v:.1f}%" for v in file_stats[(
                                'overall_quality', 'mean')]],
                            file_stats[('overall_quality', 'count')],
                            [f"{v:.1f}%" for v in file_stats[(
                                'content_fidelity_f1', 'mean')]],
                            [f"{v:.1f}%" for v in file_stats[(
                                'medical_accuracy', 'mean')]],
                            [int(v) for v in file_stats[('has_errors', 'sum')]]
                        ],
                        fill_color='white',
                        align='left',
                        height=30,
                        font=dict(size=11)
                    )
                ),
                row=1, col=1
            )

        # 2. Detailed Metrics Comparison
        detailed_metrics = {
            'Content Precision': self.data['content_fidelity_precision'].mean(),
            'Content Recall': self.data['content_fidelity_recall'].mean(),
            'Content F1': self.data['content_fidelity_f1'].mean(),
            'Medical Accuracy': self.data['medical_accuracy'].mean(),
            'Entity Coverage': self.data['entity_coverage'].mean(),
            'Section Completeness': self.data['section_completeness'].mean(),
            'Format Validity': self.data['format_validity'].mean()
        }

        colors = [self.COLORS['success'] if v >= 85 else
                  self.COLORS['warning'] if v >= 70 else
                  self.COLORS['danger'] for v in detailed_metrics.values()]

        fig.add_trace(
            go.Bar(
                x=list(detailed_metrics.keys()),
                y=list(detailed_metrics.values()),
                marker_color=colors,
                text=[f"{v:.1f}%" for v in detailed_metrics.values()],
                textposition='outside',
                textfont=dict(size=12),
                showlegend=False
            ),
            row=2, col=1
        )

        # 3. Issue Analysis
        issues = {
            'Missed Critical Info': self.data['missed_critical'].sum(),
            'Unsupported Content': self.data['unsupported_content'].sum(),
            'Processing Errors': int(self.data['has_errors'].sum()),
            'Low Quality (<70%)': int((self.data['overall_quality'] < 70).sum())
        }

        fig.add_trace(
            go.Bar(
                x=list(issues.keys()),
                y=list(issues.values()),
                marker_color=[self.COLORS['danger'], self.COLORS['warning'],
                              self.COLORS['secondary'], '#9CA3AF'],
                text=list(issues.values()),
                textposition='outside',
                textfont=dict(size=12),
                showlegend=False
            ),
            row=3, col=1
        )

        # Layout
        fig.update_layout(
            title={
                'text': f"<b>Quality Report</b><br><sub style='color:#6B7280'>{datetime.now().strftime('%B %d, %Y')}</sub>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': self.COLORS['dark']}
            },
            height=1100,
            plot_bgcolor='white',
            paper_bgcolor='#F9FAFB',
            font=dict(family="Inter, system-ui, sans-serif", size=11),
            margin=dict(l=60, r=60, t=100, b=60)
        )

        # Update axes
        fig.update_xaxes(showgrid=False, showline=True,
                         linewidth=1, linecolor='#E5E7EB')
        fig.update_yaxes(showgrid=True, gridcolor='#F3F4F6',
                         showline=True, linewidth=1, linecolor='#E5E7EB')
        fig.update_yaxes(range=[0, 105], row=2, col=1)
        fig.update_yaxes(title_text="Score (%)", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=3, col=1)

        # Save
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        fig.write_html(output_file)

        return output_file


def create_dashboard_from_files(results_files: List[str],
                                output_dir: str = "results",
                                dashboard_title: str = "SOAP Evaluation Dashboard") -> Dict[str, str]:
    """Create dashboard from results files."""
    dashboard = SOAPEvaluationDashboard(results_files, dashboard_title)

    created_files = {}

    main_dashboard = dashboard.create_comprehensive_dashboard(
        f"{output_dir}/dashboard.html")
    if main_dashboard:
        created_files['dashboard'] = main_dashboard

    quality_report = dashboard.create_quality_report(
        f"{output_dir}/quality_report.html")
    if quality_report:
        created_files['report'] = quality_report

    return created_files


def create_dashboard_cli(results_files: List[str],
                         output_file: str = "results/dashboard.html") -> str:
    """Simple CLI function for creating dashboard."""
    dashboard = SOAPEvaluationDashboard(results_files)
    return dashboard.create_comprehensive_dashboard(output_file)
