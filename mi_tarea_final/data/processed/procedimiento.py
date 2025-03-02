import pandas as pd 

datos = {
    'Nombre': ['Ana','Cris','Aldo','Luis', 'Bernie', 'Julia'],
    'Edad': [28, 25, 32, 45, 35, 28],
    'Profesion': ['Big Data', 'ML', 'Diseño', 'Mecanica', 'Ensamble', 'Diseño Grafico']
}

df = pd.DataFrame(datos)

df.to_csv('archivo_ejemplo.csv', index = False)