import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def animate(fig, duration_val, x_range, y_range, z_range):
  fig.update_layout(
    scene=dict(
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range),
            zaxis=dict(range=z_range),
        ),
    updatemenus=[{
        "buttons": [
            { "args": [None, {
                "frame": {"duration": duration_val},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration_val, "easing": "linear"},
                "repeat": True
                }],
                "label": "Play",
                "method": "animate" },
            { "args": [[None], {
                  "frame": {"duration": 0},
                  "mode": "immediate",
                  "fromcurrent": True,
                  "transition": {"duration": 0, "easing": "linear"},
                  }],
                "label": "Pause",
                "method": "animate" }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 70},
        "type": "buttons",
        "x": 0.1, "y": 0,
        "bgcolor": "white",
        "font": dict(color="black"),
        "bordercolor": "black",

    }],
      sliders = [
    {"pad": {"b": 10, "t": 30},
      "len": 0.9,
      "x": 0.1,
      "y": 0,
      "steps": [
          {"args": [[f.name], {
            "frame": {"duration": 0},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": 0, "easing": "linear"},
            }],
          "label": str(k),
          "method": "animate",
          } for k, f in enumerate(fig.frames)
      ]
    }
  ],
    template='plotly_dark',
    margin=dict(t=0, b=50, l=0, r=0),
    scene_camera=dict(
            eye=dict(x=1.6, y=1.6, z=1.6),  # Adjust eye position for zoom and angle
            center=dict(x=0, y=0, z=0)  # Center the camera view
        )
  )

  return fig
