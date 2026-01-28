import requests


def geolocalizar(direccion):
    url = "https://nominatim.openstreetmap.org/search"

    params = {
        "q": direccion,
        "format": "json",
        "limit": 1
    }

    headers = {
        "User-Agent": "mi-app-python"  # obligatorio
    }

    response = requests.get(url, params=params, headers=headers)
    data = response.json()

    if data:
        lat = data[0]["lat"]
        lon = data[0]["lon"]
        return lat, lon
    else:
        return None


# PRUEBA
resultado = geolocalizar("Quito Ecuador")
print(resultado)
