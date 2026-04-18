from dash import Dash, html

app = Dash(__name__)
app.layout = html.Div("ETHUSD agent dashboard setup is working")

if __name__ == "__main__":
    app.run(debug=True)