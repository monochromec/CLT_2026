#!/usr/bin/env python3
"""
GitHub Issue Trend Visualiser (Multi-Page App)
Reads the compressed JSON cache produced by github_issue_stats.py and renders
3-D bar charts.

Routes:
    /issues -> Issue counts (Year x Month x Count)
    /ctime  -> Avg closing time (Year x Month x Avg Days)
"""

import argparse
import gzip
import json
import pathlib
import sys

import dash
import dash_bootstrap_components as dbc
import flask
import pandas as pd
import plotly.graph_objects as go
import plotly.colors


MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    """
    Parse command-line arguments for server configuration.
    1. Define default cache file path
    2. Set up host and port options
    """
    # 1. Initialize argument parser
    # Create parser with descriptive help text
    parser = argparse.ArgumentParser(description="3-D visualisation of GitHub issues.")
    
    # 2. Add arguments
    # Configure path, port, host, and debug flag overrides
    parser.add_argument("--cache", default="issues.json.gz", metavar="PATH")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--debug", action="store_true")
    
    return parser.parse_args()


# ── Data Handling ─────────────────────────────────────────────────────────────

class IssueDataLoader:
    """Handles loading, parsing, and pivoting of GitHub issue data."""

    def __init__(self, cache_path: str):
        """
        Initialize the data loader and trigger parsing.
        :parm cache_path: Filepath to the compressed JSON data
        :type cache_path: string
        """
        self.cache_path = pathlib.Path(cache_path)
        self.df, self.df_ctime, self.total_issues = self._load_data()

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, int]:
        """
        Load and process issue data from the JSON cache.
        1. Read and decompress file
        2. Extract timestamps for creation and closure
        3. Calculate time-to-close durations
        """
        # 1. Validate file existence
        # Exit script immediately if the specified cache is missing
        if not self.cache_path.exists():
            sys.exit(f"Error: cache file not found: {self.cache_path}")

        # 2. Read cache contents
        # Open gzip file in binary mode and decode JSON structure
        with gzip.open(self.cache_path, "rb") as fh:
            issues = json.loads(fh.read().decode())

        rows = []
        ctime_rows = []

        # 3. Process individual issues
        # Iterate over JSON nodes to build structured flat lists
        for issue in issues:
            
            # Extract and parse timestamps to UTC, handling nulls gracefully
            dt_created = pd.to_datetime(issue.get("created_at"), utc=True) if issue.get("created_at") else None
            dt_closed = pd.to_datetime(issue.get("closed_at"), utc=True) if issue.get("closed_at") else None

            # Append opening event
            if dt_created:
                rows.append({"event": "created_at", "year": dt_created.year, "month": dt_created.month, "number": issue["number"]})
            
            # Append closing event
            if dt_closed:
                rows.append({"event": "closed_at", "year": dt_closed.year, "month": dt_closed.month, "number": issue["number"]})
            
            # 4. Calculate closing duration
            # Only calculate if both creation and closing timestamps exist
            if dt_created and dt_closed:
                # Convert total seconds between timestamps into fractional days
                days_to_close = (dt_closed - dt_created).total_seconds() / 86400.0
                if days_to_close >= 0:
                    ctime_rows.append({"year": dt_closed.year, "month": dt_closed.month, "days": days_to_close})

        # Return structured DataFrames alongside the raw issue count
        return pd.DataFrame(rows), pd.DataFrame(ctime_rows), len(issues)

    def get_pivot(self, event: str) -> pd.DataFrame:
        """
        Create a year-by-month matrix of issue counts.
        :parm event: The timestamp field to pivot on ('created_at' or 'closed_at')
        :type event: string
        """
        # 1. Filter subset
        # Isolate rows matching the requested event type
        sub = self.df[self.df["event"] == event].copy()
        
        # 2. Aggregate and pivot
        # Group by year and month, count issues, and reshape into a 2D matrix
        counts = sub.groupby(["year", "month"])["number"].count().reset_index(name="count")
        pivot = counts.pivot(index="year", columns="month", values="count").fillna(0)
        
        return self._ensure_all_months(pivot)

    def get_ctime_pivot(self) -> pd.DataFrame:
        """
        Create a year-by-month matrix of average closing times.
        """
        # 1. Handle empty states
        # Return empty DataFrame if no closing data exists to prevent errors
        if self.df_ctime.empty:
            return pd.DataFrame()
            
        # 2. Aggregate averages
        # Group by year and month and calculate the mean duration in days
        avg_times = self.df_ctime.groupby(["year", "month"])["days"].mean().reset_index(name="avg_days")
        
        # Reshape into a 2D matrix
        pivot = avg_times.pivot(index="year", columns="month", values="avg_days").fillna(0)
        return self._ensure_all_months(pivot)

    def _ensure_all_months(self, pivot: pd.DataFrame) -> pd.DataFrame:
        """
        Pad pivot table with zeroes for missing months.
        :parm pivot: The raw pivot table to format
        :type pivot: pandas.DataFrame
        """
        # 1. Fill missing columns
        # Iterate through months 1-12 and explicitly add a 0 column if missing
        for m in range(1, 13):
            if m not in pivot.columns:
                pivot[m] = 0
                
        # 2. Sort columns
        # Ensure chronological order from Jan to Dec
        return pivot[sorted(pivot.columns)]

    @property
    def min_year(self) -> int:
        return int(self.df["year"].min())

    @property
    def max_year(self) -> int:
        return int(self.df["year"].max())


# ── Chart Rendering ───────────────────────────────────────────────────────────

class ChartBuilder:
    """Utility class for constructing Plotly 3-D figures."""

    @staticmethod
    def _create_bar_mesh(x_centre, y_centre, z_top, color, opacity, hover_text) -> go.Mesh3d:
        """
        Construct a 3-D rectangular prism using Plotly's Mesh3d component.
        :parm x_centre: Center coordinate on the X-axis (Year)
        :type x_centre: integer
        :parm y_centre: Center coordinate on the Y-axis (Month)
        :type y_centre: integer
        :parm z_top: Height coordinate on the Z-axis (Value)
        :type z_top: float
        :parm color: Plotly-compatible color string/hex for the bar
        :type color: string
        :parm opacity: Alpha transparency value between 0.0 and 1.0
        :type opacity: float
        :parm hover_text: Formatted HTML string for mouseover tooltip
        :type hover_text: string
        """
        # 1. Define dimensions
        # Set base width/depth to 0.6 units to leave gaps between bars
        dx, dy = 0.6, 0.6
        x0, x1 = x_centre - dx / 2, x_centre + dx / 2
        y0, y1 = y_centre - dy / 2, y_centre + dy / 2
        z0, z1 = 0, z_top

        # 2. Define vertices
        # Generate the 8 corner points of the rectangular prism
        vx = [x0, x1, x1, x0, x0, x1, x1, x0]
        vy = [y0, y0, y1, y1, y0, y0, y1, y1]
        vz = [z0, z0, z0, z0, z1, z1, z1, z1]

        # 3. Define faces
        # Map vertices into 12 distinct triangular faces
        i = [0, 0, 0, 0, 4, 4, 1, 1, 2, 2, 3, 3]
        j = [1, 2, 3, 5, 5, 6, 2, 5, 3, 6, 0, 7]
        k = [2, 3, 7, 4, 6, 7, 5, 6, 6, 7, 7, 4]

        # Return constructed 3D shape
        return go.Mesh3d(
            x=vx, y=vy, z=vz, i=i, j=j, k=k,
            color=color, opacity=opacity, showscale=False,
            hovertemplate=hover_text,
        )

    @classmethod
    def build_figure(cls, pivot: pd.DataFrame, event_label: str, colorscale_name: str, opacity: float, z_label: str) -> go.Figure:
        """
        Assemble the full Plotly chart containing all 3-D bars.
        :parm pivot: Aggregated dataset to visualize
        :type pivot: pandas.DataFrame
        :parm event_label: Label for the chart title
        :type event_label: string
        :parm colorscale_name: Name of the Plotly colorscale to apply
        :type colorscale_name: string
        :parm opacity: Transparency of the bars
        :type opacity: float
        :parm z_label: Unit label for the Z-axis (e.g. 'Count' or 'Avg Days')
        :type z_label: string
        """
        years = list(pivot.index)
        months = list(range(1, 13))

        # 1. Setup Color Scaling
        # Find the maximum height to normalize color intensity across the grid
        z_max = pivot.values.max() if pivot.values.max() > 0 else 1
        scale = plotly.colors.get_colorscale(colorscale_name)

        def z_to_color(z):
            # Calculate fractional height and pull matching color from scale
            frac = float(z) / float(z_max)
            return plotly.colors.sample_colorscale(scale, frac, colortype="rgb")[0]

        # 2. Build Traces
        # Loop through every year/month cell and generate a 3D bar if value > 0
        traces = []
        for yi, year in enumerate(years):
            for month in months:
                val = pivot.loc[year, month]
                if val > 0:
                    # Format hover text dynamically based on the unit type
                    val_str = f"{val:,.1f}" if "Days" in z_label else f"{val:,.0f}"
                    hover = f"Year: {year}<br>Month: {MONTH_NAMES[month-1]}<br>{z_label}: {val_str}<extra></extra>"
                    
                    traces.append(cls._create_bar_mesh(
                        x_centre=yi, y_centre=month, z_top=val,
                        color=z_to_color(val), opacity=opacity, hover_text=hover
                    ))

        # 3. Configure Layout
        # Assemble figure and apply styling, axis titles, and camera angles
        fig = go.Figure(data=traces)
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="Year", tickvals=list(range(len(years))), ticktext=[str(y) for y in years], gridcolor="white"),
                yaxis=dict(title="Month", tickvals=list(range(1, 13)), ticktext=MONTH_NAMES, gridcolor="white"),
                zaxis=dict(title=z_label, gridcolor="white"),
                camera=dict(eye=dict(x=1.8, y=-1.8, z=1.2)),
                aspectmode="manual", aspectratio=dict(x=1.6, y=1.2, z=0.8),
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title=dict(text=event_label, font=dict(size=14)),
            showlegend=False, height=580, uirevision="static",
        )
        return fig


# ── App Orchestration ─────────────────────────────────────────────────────────

class IssueDashboardApp:
    def __init__(self, data_loader: IssueDataLoader):
        """
        Initialize the Dash application and wire dependencies.
        :parm data_loader: Instantiated data loader object
        :type data_loader: IssueDataLoader
        """
        self.data = data_loader
        self.pivot_created = self.data.get_pivot("created_at")
        self.pivot_closed = self.data.get_pivot("closed_at")
        self.pivot_ctime = self.data.get_ctime_pivot()

        # 1. Initialize Flask and Dash
        # Enable callback suppression to allow multi-page routing
        self.server = flask.Flask(__name__)
        self.app = dash.Dash(
            __name__, server=self.server,
            external_stylesheets=[dbc.themes.FLATLY],
            title="GitHub Issue Trends",
            suppress_callback_exceptions=True
        )

        # 2. Wire app components
        self._setup_layout()
        self._register_callbacks()
        self._register_routes()

    def _setup_layout(self):
        """
        Define the HTML layout and UI components of the dashboard.
        """
        # 1. Prepare dynamic layout elements
        # Calculate year ranges to determine appropriate slider tick marks
        year_span = self.data.max_year - self.data.min_year
        step = 1 if year_span <= 10 else (2 if year_span <= 20 else 5)
        slider_marks = {y: str(y) for y in range(self.data.min_year, self.data.max_year + 1) if y % step == 0}

        # 2. Build Navigation Bar
        navbar = dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Issue Counts", href="/issues", id="nav-issues")),
                dbc.NavItem(dbc.NavLink("Closing Times", href="/ctime", id="nav-ctime")),
            ],
            brand="GitHub Repository Visualiser",
            color="primary",
            dark=True,
            className="mb-3",
            style={"padding": "0.5rem"}
        )

        # 3. Build Control Panel
        # Scaled down panel utilizing Bootstrap's small constraints
        controls = dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Dataset", html_for="dd-event", style={"fontSize": "0.8rem", "marginBottom": "2px"}),
                        dbc.Select(
                            id="dd-event",
                            options=[
                                {"label": "Issues opened (created_at)", "value": "created_at"},
                                {"label": "Issues closed (closed_at)", "value": "closed_at"},
                            ],
                            value="created_at",
                            size="sm",
                        ),
                    ], md=4, id="col-dataset"), 
                    dbc.Col([
                        dbc.Label("Year filter", html_for="sl-years", style={"fontSize": "0.8rem", "marginBottom": "2px"}),
                        dash.html.Div(
                            dash.dcc.RangeSlider(
                                id="sl-years", min=self.data.min_year, max=self.data.max_year, step=1,
                                value=[self.data.min_year, self.data.max_year], marks=slider_marks,
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                            style={"transform": "scale(0.9)", "transformOrigin": "left top"}
                        )
                    ], md=4),
                    dbc.Col([
                        dbc.Label("Bar Opacity", html_for="sl-opacity", style={"fontSize": "0.8rem", "marginBottom": "2px"}),
                        dash.html.Div(
                            dash.dcc.Slider(
                                id="sl-opacity", min=0.1, max=1.0, step=0.05, value=0.55,
                                marks={0.1: '10%', 0.5: '50%', 1.0: '100%'},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                            style={"transform": "scale(0.9)", "transformOrigin": "left top"}
                        )
                    ], md=4),
                ]),
            ], style={"padding": "0.5rem"}), 
            className="mb-2 shadow-sm",
        )

        # 4. Assemble Application Container
        # Inject navigation, controls, 3D graph component, and text footer
        self.app.layout = dash.html.Div([
            dash.dcc.Location(id="url", refresh=False),
            navbar,
            dbc.Container(fluid=True, children=[
                dbc.Row(dbc.Col(
                    dbc.Card(dbc.CardBody([
                        dash.html.H5(id="page-title", className="card-title mb-0", style={"fontSize": "1.1rem", "fontWeight": "bold"}),
                        dash.html.P(f"{self.data.total_issues:,} issues available", className="text-muted mb-0", style={"fontSize": "0.75rem"}),
                    ], style={"padding": "0.5rem"}), className="mb-2 shadow-sm bg-light text-dark"), width=12
                )),
                dbc.Row(dbc.Col(controls, width=12)),
                dbc.Row(dbc.Col(
                    dash.dcc.Graph(id="chart-3d", config={"displayModeBar": True, "scrollZoom": True}, style={"height": "550px"}),
                    width=12,
                )),
                dbc.Row(dbc.Col(dash.html.Div(id="stats-bar", className="text-muted small text-center mb-2"), width=12)),
            ])
        ])

    def _register_callbacks(self):
        """
        Bind reactive functions to UI state changes.
        """
        @self.app.callback(
            dash.Output("col-dataset", "style"),
            dash.Output("page-title", "children"),
            dash.Output("nav-issues", "active"), 
            dash.Output("nav-ctime", "active"),
            dash.Input("url", "pathname")
        )
        def toggle_ui(pathname):
            """
            Handle layout shifts based on current URL path.
            1. Evaluate path and adjust visibility
            2. Update navigation active states
            :parm pathname: The current URL path segment
            :type pathname: string
            """
            # 1. Closing Time Route
            if pathname == '/ctime':
                return {"display": "none"}, "Average Closing Time", False, True
            
            # 2. Default Issue Route
            return {}, "Issue Volume", True, False

        @self.app.callback(
            dash.Output("chart-3d", "figure"),
            dash.Output("stats-bar", "children"),
            dash.Input("url", "pathname"),
            dash.Input("dd-event", "value"),
            dash.Input("sl-years", "value"),
            dash.Input("sl-opacity", "value"),
        )
        def update_chart(pathname, event, year_range, opacity):
            """
            Re-render the 3-D figure and summary statistics based on inputs.
            :parm pathname: The active page route
            :type pathname: string
            :parm event: The dataset dropdown value
            :type event: string
            :parm year_range: A two-item list containing [min_year, max_year]
            :type year_range: list
            :parm opacity: Transparency float from the slider
            :type opacity: float
            """
            y0, y1 = int(year_range[0]), int(year_range[1])
            n_years = y1 - y0 + 1

            # 1. Determine active dataset and styling overrides based on route
            if pathname == '/ctime':
                pivot = self.pivot_ctime
                event_label = "Average Time to Close"
                z_label = "Avg Days"
                colorscale = "Plasma"
            else:
                pivot = self.pivot_created if event == "created_at" else self.pivot_closed
                event_label = "Issues Opened per Month" if event == "created_at" else "Issues Closed per Month"
                z_label = "Count"
                colorscale = "Teal"

            # 2. Slice data by timeframe
            pivot_filtered = pivot.loc[(pivot.index >= y0) & (pivot.index <= y1)]

            # Short-circuit rendering if subset is empty
            if pivot_filtered.empty or pivot_filtered.values.sum() == 0:
                fig = go.Figure().update_layout(
                    annotations=[dict(text="No data for selected range", showarrow=False, font=dict(size=18), xref="paper", yref="paper", x=0.5, y=0.5)],
                    height=580
                )
                return fig, ""

            # 3. Request visualization rendering
            fig = ChartBuilder.build_figure(pivot_filtered, event_label, colorscale, opacity, z_label)

            # 4. Compute aggregate statistics for the footer
            peak_idx = pivot_filtered.values.argmax()
            peak_year = pivot_filtered.index[peak_idx // 12]
            peak_month = MONTH_NAMES[(peak_idx % 12)]
            peak_val = pivot_filtered.values.max()

            if pathname == '/ctime':
                avg_val = pivot_filtered.values[pivot_filtered.values > 0].mean()
                stats = f"Showing data over {n_years} year(s) · Overall average: {avg_val:,.1f} days to close · Peak delay: {peak_val:,.1f} days in {peak_month} {peak_year}"
            else:
                total = int(pivot_filtered.values.sum())
                avg_per_year = total / n_years if n_years else 0
                stats = f"Showing {total:,} events over {n_years} year(s) · avg {avg_per_year:,.0f}/year · peak: {peak_month} {peak_year}"
            
            return fig, stats

    def _register_routes(self):
        """
        Configure Flask-level route redirects.
        """
        @self.server.route("/")
        def root_redirect():
            # 1. Force absolute redirect
            # Ensure visitors to the root domain land on the explicit dashboard path
            return flask.redirect("/issues")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    """
    Script entrypoint: Initialize components and start webserver.
    """
    args = parse_args()
    data_loader = IssueDataLoader(args.cache)
    app = IssueDashboardApp(data_loader)
    
    print(f"Server starting.")
    print(f" -> Issues Dashboard: http://{args.host}:{args.port}/issues")
    print(f" -> Closing Time Dashboard: http://{args.host}:{args.port}/ctime")
    
    app.server.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
    
