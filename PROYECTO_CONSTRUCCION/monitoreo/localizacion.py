import requests


def geolocalizar(lat, lon):
    url = "https://nominatim.openstreetmap.org/search"

    params = {
        "lat": lat,
        "lon": lon,
        "format": "json",
    }

    headers = {
        "User-Agent": "mi-app-python"
    }

    response = requests.get(url, params=params, headers=headers)
    data = response.json()




# PRUEBA
resultado = geolocalizar()
print(resultado)
