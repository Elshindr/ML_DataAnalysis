from bokeh.io import output_notebook, show
from bokeh.plotting import figure
output_notebook()

import pandas as pd
import numpy as np

df=pd.read_csv('flights.csv', index_col=0)
df.head()



def to_weekday(val):
    return pd.to_datetime(val).weekday()
df["weekday"]=df["time_hour"].apply(to_weekday)

lst_travel_day = df["weekday"].value_counts()
print("Nombre total de vols pour chaque jour de la semaine", lst_travel_day)



lst_days= ["Mon", "Tue", "Wed", "Thu", "Fr", "Sat", "Sun"]

f = figure(plot_width=600, plot_height=400, x_range=lst_days, title="Nombre de vols effectués pour chaque jour de la semaine", y_range=[1000, 60000], y_axis_label= "nombre de vols")
f.vbar(x=lst_days, top = lst_travel_day, width=0.5 )

show(f)

:

delay_united = df['arr_delay'][df['name'] == "United Air Lines Inc."]
delay_express = df['arr_delay'][df['name'] == "ExpressJet Airlines Inc."]
delay_jetblue = df['arr_delay'][df['name'] == "JetBlue Airways"]


p_united = figure(plot_width=600, plot_height=400, title="Retard par compagnie", y_range=[0, 0.03], y_axis_label= "Fréquence", x_axis_label="Retard en minutes")
p_express= figure(plot_width=600, plot_height=400, title="Retard par compagnie", y_range=[0, 0.03], y_axis_label= "Fréquence", x_axis_label="Retard en minutes")
p_jetblue= figure(plot_width=600, plot_height=400, title="Retard par compagnie", y_range=[0, 0.03], y_axis_label= "Fréquence", x_axis_label="Retard en minutes")

hist_united, edges_united   = np.histogram(a=delay_united, density=True, bins=50, range=[-60, 1120])
hist_express, edges_express = np.histogram(a=delay_express, density=True, bins=50, range=[-60, 1120])
hist_jetblue, edges_jetblue = np.histogram(a=delay_jetblue, density=True, bins=50, range=[-60, 1120])

from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models.annotations import Span

source_united = ColumnDataSource({'hist': hist_united, 'edges':edges_united[:-1]})
source_express= ColumnDataSource({'hist': hist_express, 'edges':edges_express[:-1]})
source_jetblue= ColumnDataSource({'hist': hist_jetblue, 'edges':edges_jetblue[:-1]})

p_united.vbar(x='edges', top='hist', width=3, color='red', source=source_united)
p_express.vbar(x='edges', top='hist', width=3, color='teal', source=source_express)
p_jetblue.vbar(x='edges', top='hist', width=3, color='orange', source=source_jetblue)


hover = HoverTool(tooltips = [("retard", "@edges"), ("fréquence", "@hist")])
p_united.add_tools(hover)
p_express.add_tools(hover)
p_jetblue.add_tools(hover)

straw = Span(dimension='height', location=0, line_color= 'purple')
p_united.add_layout(straw)
p_express.add_layout(straw)
p_jetblue.add_layout(straw)

from bokeh.models.widgets import Panel, Tabs

tab_united = Panel(child = p_united, title='United Air Lines Inc.')
tab_express = Panel(child = p_express, title='ExpressJet Airlines Inc.')
tab_jetblue = Panel(child = p_jetblue, title='JetBlue Airways')

tabs = Tabs(tabs= [tab_united,tab_express,tab_jetblue])
show(tabs)

# A premiere vue, la compagnie 'ExpressJet Airlines Inc.' aurait plus de retard longs que les autres avec par exemple 176min de retard pour une fréquence de 2.6.10^-4
# Enfin, United Air Lines Inc. serait la compagnie qui aurait le plus tendance à arriver en avance