- Details about stations are in file: AWS_details.csv
	AWS - station code
	Region		
	Punkt	
	Location	
	Plant	- plants where the station is plased
	start	- start of the measurements
	Comment	- coment on data or moving of the station
	end	- end of the measurements
	year_start	
	year_end	
	lon	- logitude
	lat	- latitude
	elev - elevation


- Daily meteorology is in one file: daily_data_multiplestations.csv 
data are from 2013 to 2021-12-07 (collected)
	AWS - station code
	Date - date
	Year 
	Tavg - daily average temperature [°C]
	Tmax - daily maximum temperature [°C]
	Tmin - daily minimum temperature [°C]	
	Prec - precipitation mm
	Vl - leaf wetness
	Rh - relative humidity %
	Tzavg - daily average soil temperature [°C]
	Tzmax - daily maximum soil temperature [°C]
	Tzmin - daily minimum soil temperature [°C]

- Hourly data are in separate files which are named with station code:
data are from 2013 to 2021-12-31 (collected)	
	Datum/ vrijeme - day/time
	HC Air temperature [°C]		
		max	
		min	
		average
	Dew Point [°C]		
		min	
		average	
	HC Relative humidity [%]
		max	
		min	
		average			
	Precipitation [mm]	
	Leaf Wetness [min]	
	Battery voltage [mV]

