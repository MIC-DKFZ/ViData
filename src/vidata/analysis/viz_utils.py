import plotly.io as pio


def save_figure(figure, path, as_thml=True):
    pio.write_image(figure, file=str(path) + ".png", scale=3)
    if as_thml:
        figure.write_html(str(path) + ".html", include_plotlyjs="cdn", full_html=True)


def adjust_layout(
    figure, title, xaxis_title=None, yaxis_title=None, width=1000, height=750, subplots=False
):
    figure.update_layout(
        title={"text": f"<b>{title}</b>", "font": {"size": 24}},
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 80, "b": 40},
        width=width,
        height=height,
    )
    if xaxis_title is not None:
        figure.update_layout(
            xaxis={
                "title": {"text": xaxis_title, "font": {"size": 18}},  # axis label font
                # "tickfont": {"size": 14},  # tick labels
            }
        )
    else:
        for axis_name in figure.layout:
            if axis_name.startswith("xaxis"):
                figure.layout[axis_name].title.font.size = 18
                figure.layout[axis_name].tickfont.size = 14

    if yaxis_title is not None:
        figure.update_layout(
            yaxis={
                "title": {"text": yaxis_title, "font": {"size": 18}},  # axis label font
                "tickfont": {"size": 14},  # tick labels
            }
        )
    else:
        for axis_name in figure.layout:
            if axis_name.startswith("yaxis"):
                figure.layout[axis_name].title.font.size = 18
                figure.layout[axis_name].tickfont.size = 14

    if subplots:
        for annotation in figure.layout.annotations:
            annotation.font.size = 18
