# -*- coding: utf-8 -*-
"""
Created on Thu May 28 22:00:12 2020

@author: Mathew
"""

from multi_rake import Rake

text_en = ("Our project aims at predicting floods in various susceptible coastal areas. The risk of coastal flooding is increasing as a result of deforestation, untimely and torrential rains, sea level rise etc. In the event of floods occurring, a large amount of damage can be curbed if we have a model that could predict floods in advance. To avoid devastating consequences which have a major impact on economic and financial conditions of the city, our system aims to provide maximum accuracy in alerting target regions about the forthcoming disasters. This model can be extended for different locations in the future.One of the major factors causing floods is heavy rainfall, along with catchment and weather conditions before rainfall, tidal influences, inadequate drainage system, lack of reservoirs and catchment areas, deforestation, etc. With this model of predicting floods because of deforestation based on ML will provide clear and efficient output of forecasting the flood in the coastal regions. The model will take s from satellite, classify s based on the factors required and process the input dataset.We will use statistical data based on various factors, maps and s. The datasets can either be real-time data collected from Synthetic Aperture Radar(SAR) or historical data recorded before.")

rake = Rake()

keywords = rake.apply(text_en)

print(keywords[:10])
