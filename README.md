# Calgary Plus 15 Map & Transit Dashboard

### View Here
[https://iyanvelji-plus-15-map-project.share.connect.posit.cloud/](https://iyanvelji-plus-15-map-project.share.connect.posit.cloud/)

## 1. Calgary Plus 15 Map

An interactive map of Calgary's Plus 15 pedestrian network.

**Route calculation**
- Find a route through the Plus 15 network. Enter a start point and end point and click "Find Route". Select points on the plus 15 map to see building names. 

**Notes**
- Building-to-street connections are nearest-street approximations; they are not verified building entrance locations.
- Some Plus 15 connections may be closed or inaccessible.
- The map is for general reference and is not a substitute for official City of Calgary maps.
- The routing network is limited to the downtown area.

## 2. Calgary Transit Dashboard

**Visualizations**

- 2025 C-Train Boardings
- Transit’s Portion of the Budget
- Transit Ridership by Fare Category
- C-Train and Bus Ridership
- Youth vs Senior Ridership
- Fare Type Popularity
- Monthly Ridership Seasonality
- Annual Transit Ridership

## Technologies Used

- Python
- Shiny for Python
- Pandas and NumPy
- ipyleaflet
- Bokeh
- Highcharts

## Run locally

**Requirements**

Python 3.10 or newer

The Python packages listed in requirements.txt

The project CSV data files

[https://iyanvelji-plus-15-map-project.share.connect.posit.cloud/](https://iyanvelji-plus-15-map-project.share.connect.posit.cloud/)

# Calgary Plus 15 Map & Transit Dashboard

An interactive Python Shiny application featuring two Calgary-focused tools: a downtown Plus 15 skywalk mapping and walking-route experience, and a Calgary Transit dashboard with ridership, station boarding, and budget visualizations.

## App overview

The application is organized into two main sections, accessible through the tabs at the top of the app.

### 1. Calgary Plus 15 Map

Explore Calgary's downtown Plus 15 pedestrian network on an interactive map.

**Features**

- **Building and address search:** Enter a Plus 15 building name or a Calgary street address as the starting point and destination.
- **Walking-route calculation:** Find a route through the combined downtown street and Plus 15 network. The app uses a graph representation of street segments and skywalk connections and calculates paths with Dijkstra's shortest-path algorithm.
- **Route highlighting:** View Plus 15 portions of a selected route separately from street-walking and building-to-street connector segments.
- **Building hubs and popups:** Select building hubs or route segments on the map to view building names. Start and destination locations are marked separately.
- **Map styling:** The map uses OpenStreetMap tiles with a darkened map appearance and a gold-toned style for the Plus 15 network.

**Routing notes and limitations**

- Building-to-street connections are nearest-street approximations; they are not verified building entrance locations.
- Some Plus 15 connections may be closed or inaccessible. The map is for general reference and is not a substitute for official City of Calgary maps or current access information.
- Address locations may be interpolated between nearby parcel addresses when an exact parcel record is unavailable.
- The routing network is limited to the downtown area covered by the data included with the project.

### 2. Calgary Transit Dashboard

Explore transit usage and budget information through interactive maps and charts.

**Visualizations**

- **2025 C-Train Boardings:** A map of C-Train stations. Circle size and colour communicate boarding volume, and selecting a station opens its boarding total.
- **Transit’s Portion of the Budget:** An interactive 3D pie chart showing budget by business unit. Use the slider to change the number of displayed slices; smaller categories are grouped into an “Other” slice. Calgary Transit and the Green Line are highlighted.
- **Transit Ridership by Fare Category:** A doughnut chart comparing adult, youth, U-Pass, and seniors pass ridership for the latest year available in the loaded ridership data.
- **C-Train and Bus Ridership:** A time-series view of monthly C-Train and bus ridership, with a month-range filter.
- **Youth vs Senior Ridership:** A comparison of youth and senior ridership over a selectable year range.
- **Fare Type Popularity:** A bump chart showing how fare categories rank over time, with a year-range filter.
- **Monthly Ridership Seasonality:** A seasonal comparison of monthly ridership patterns across years.
- **Annual Transit Ridership:** An annual ridership chart with a year-range filter.

The dashboard uses a dark visual theme and interactive chart controls to support exploration of the data.

## Technology

- **Python** — application logic and data processing
- **Shiny for Python** — app layout, inputs, and reactive outputs
- **Pandas and NumPy** — data loading and analysis
- **ipyleaflet and ipywidgets** — interactive Plus 15 map components
- **Bokeh** — annual ridership visualization
- **Highcharts and Leaflet (JavaScript)** — embedded dashboard charts and the C-Train station map
