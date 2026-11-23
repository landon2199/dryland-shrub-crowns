# Literature review: shrub crowns in a desert shrubland, and what Sentinel-2 can see

This review supports the `dryland-shrub-crowns` project. The project counts and measures individual shrub crowns on sub-metre imagery at the NEON Jornada site (JORN) in the northern Chihuahuan Desert, New Mexico. The shrubs are mainly honey mesquite, creosotebush and tarbush, growing on bare soil. Crowns are mapped on USDA NAIP 0.6 m four-band imagery (May 2022). A 0.5 m canopy height model (CHM) from NEON airborne lidar (September 2022, about 35 points per m2) is the reference. The project then asks a second question: how much of this sparse shrub cover can Sentinel-2 L2A see at 10 m? A short add-on applies the same index thinking to saxaul planting on the dried Aral Sea bed.

Each claim below comes from the abstract or text of the cited paper. Where a source gives a number, the number is the source's. DOIs were checked against Crossref or DataCite (see the table at the end).

## 1. Crown detection in drylands from very high resolution imagery

**Classic methods.** The older approach is to segment the image and then classify the segments. This is object-based image analysis (OBIA). Laliberte et al. (2004) used multi-resolution segmentation and object-oriented classification on 11 historical aerial photos (1937 to 1996) and a 2003 QuickBird image of a 150 ha area of the Jornada Basin. Their method found 87% of shrubs larger than 2 m2. They also noted that pixel size made shrub and grass cover come out too low. Laliberte and Rango (2009) applied OBIA to sub-decimetre UAV imagery of rangeland. Adding texture measures raised classification accuracy at nearly all segmentation scales, and accuracy was highest at coarser scales.

Karlson et al. (2014) mapped tree crowns in semi-arid West Africa with OBIA on WorldView-2 imagery. They detected 85.4% of individual crowns and crown clusters. Delineation error (the difference between mapped and field-measured crown area) was 45.6% for single crowns and 61.5% for clusters. Accuracy dropped where trees were dense and where understory was dense, and it was better for medium and large crowns than for small ones. This is a useful warning for any crown-area claim: detecting a crown is easier than outlining it well.

The project's baseline (an MSAVI or NDVI threshold followed by a watershed split) follows the same segment-then-decide logic as these studies. Its value is that it is cheap, transparent and needs no training data.

**Deep learning.** Brandt et al. (2020) mapped the crown size of every tree larger than 3 m2 across 1.3 million km2 of the West African Sahara, Sahel and sub-humid zone, using sub-metre satellite imagery and deep learning. They counted over 1.8 billion trees (13.4 per hectare), with a median crown of 12 m2. Mapped canopy cover rose from 0.1% in hyper-arid areas to 13.3% in sub-humid areas. This shows that crown-by-crown mapping of isolated dryland woody plants works at very large scale.

Weinstein et al. (2019) trained a crown detector on RGB imagery at an open-canopy NEON site. They first used lidar-based unsupervised delineation to make noisy training crowns, then refined with a small hand-labelled set. At an intersection-over-union (IoU) threshold of 0.5, recall was 0.69 and precision was 0.61. Weinstein et al. (2020a) released this approach as the DeepForest Python package. Weinstein et al. (2020b) found that models pretrained on other sites and fine-tuned on a small local set performed like local models, and that performance was best in open oak woodlands. For shrubs, Guirado et al. (2021) compared OBIA and Mask R-CNN (He et al., 2017) on scattered *Ziziphus lotus* shrubs in a dryland. Fusing the two raised segmentation accuracy by up to 25% over either alone.

**Segment Anything Model (SAM).** Kirillov et al. (2023) introduced SAM as a promptable segmentation model. It was trained on a dataset of over 1 billion masks from 11 million images, and it generalises zero-shot to new image types. Osco et al. (2023) tested SAM on remote sensing images with point, box and text prompts. They report that SAM is promising for aerial and orbital images, and that it struggles with lower spatial resolution. Wu and Osco (2023) released `samgeo`, a Python package that runs SAM on geospatial rasters automatically, interactively or with text prompts, and exports vector and raster results. Two SAM modes are therefore natural for this project: automatic masks, and point prompts placed on candidate crowns. The literature reviewed here gives no verified SAM result for sparse desert shrubs on 0.6 m four-band imagery. That is a test this project can run.

## 2. Jornada and Chihuahuan Desert shrub mapping

The Jornada Basin has a long mapping record because it changed from grassland to shrubland. Laliberte et al. (2004) measured the change over 66 years: shrub cover rose from about 0.9% in 1937 to 13.1% in 2003, and grass cover fell from about 18.5% to 1.9%. Rango et al. (2009) argued that hyperspatial UAV imagery allows plant cover, composition and structure to be measured at several scales. Laliberte et al. (2011) built a species-level rangeland classification from UAS multispectral imagery with 87% overall accuracy, and compared its reflectance with WorldView-2 data.

Ji et al. (2019) made a shrub cover map of the Jornada Basin LTER site from high spatial resolution images and used it to study limits on shrub cover. Their results link maximum shrub cover to plant-available water and surface clay content, and the cover-density relationship suggests self-thinning in dense stands. A derived dataset (Roberts & Hanan, 2023) combines the Ji et al. map (the dataset title names NAIP 2011) with USGS lidar shrub heights from 2019. It gives shrub volume, cover, height and density at 1 ha resolution. That grid is too coarse to check individual crowns, which is why this project builds its own crown reference from the 2022 NEON point cloud.

Lidar has been used on Jornada shrubs before. Rango et al. (2000) used scanning lidar to characterise the morphology of shrub coppice dunes in desert grassland in southern New Mexico. Outside Jornada, Mitchell et al. (2011) compared small-footprint lidar (9.46 points per m2) with field-measured sagebrush. Lidar predicted shrub height well (R2 0.84 to 0.86) but under-estimated true height by about 30%. It also predicted canopy area (R2 0.78). Two lessons carry over. A CHM is a good reference for crown presence, position and relative size. Its heights are biased low for shrubs, so absolute height claims need care. The project's point density (about 35 per m2) is higher than in that study, but the bias should still be reported, not assumed away.

## 3. Vegetation indices over bright soil, and the limits of 10 m pixels

**The indices.** NDVI comes from the red and near-infrared bands. Rouse et al. (1974) is the usual citation for the normalised difference index, and Tucker (1979) studied red and photographic infrared linear combinations for monitoring vegetation. NDVI is sensitive to the soil under sparse plants. Huete (1988) proposed the soil-adjusted vegetation index (SAVI), which adds a soil adjustment factor to NDVI. Qi et al. (1994) proposed the modified SAVI (MSAVI), which replaces the fixed factor with a variable function. They report that MSAVI increases the dynamic range of the vegetation signal while reducing soil background influence. Over bright desert soil with sparse shrubs, this is the reason to test MSAVI next to NDVI.

**Cover and detection limits.** Several studies show that a single index is a weak measure of sparse cover. Smith et al. (1990) used spectral mixture analysis on Landsat TM images of Owens Valley, California, to estimate vegetation, soil and shade fractions in 30 m pixels. They checked the fractions against ground transects, and concluded the approach could map and monitor sparse vegetation over large regions. Elmore et al. (2000) compared spectral mixture analysis with NDVI against six years of field data at 33 Owens Valley sites. Mixture analysis estimated live cover to within about 4% cover. NDVI correlated with live cover but was not a quantitative measure, and it gave the correct direction of change in only 67% of samples.

Okin et al. (2001) simulated AVIRIS hyperspectral data for arid and semi-arid areas. In their simulation, vegetation type could not be reliably retrieved below about 30% cover, although cover itself could often be estimated. Their finding is about type discrimination with hyperspectral data, not shrub detection with Sentinel-2. It does show that cover thresholds in drylands are real and are usually stated in percent cover.

At continental scale, Jones et al. (2018) and Allred et al. (2021) estimate fractional cover of plant types, including shrubs, on western U.S. rangelands from Landsat and field plots. Allred et al. report a mean absolute error of 6.3% and a root mean squared error of 9.6% on held-out plots. For sparse shrublands, an error of that size is large compared with the cover itself. Sentinel-2 (Drusch et al., 2012) adds 10 m pixels, but a 10 m pixel is still about 280 times the area of a 0.6 m NAIP pixel (100 m2 against 0.36 m2).

None of the sources reviewed here gives a single cover threshold below which Sentinel-2 indices stop seeing shrubs. They give error sizes and unmixing results, not a detection limit.

## 4. Validation: matching crowns, and honest error under spatial dependence

**Object matching.** Crown detection is scored by overlap between mapped and reference crowns. IoU is the standard overlap score in object-detection benchmarks such as the PASCAL VOC challenge (Everingham et al., 2010). A mapped crown counts as a match above an IoU threshold, and precision, recall and F1 follow from the match counts. Weinstein et al. (2019) report precision and recall at an IoU of 0.5. Karlson et al. (2014) separate detection rate from delineation error. This project reports both: match-based F1 at fixed IoU thresholds, and the error in crown area for matched crowns.

**Spatial cross-validation.** Neighbouring locations are not independent, so random splits make a model look better than it is. Roberts et al. (2017) review this and show that block cross-validation is nearly always more appropriate than random cross-validation when predicting to new data. Ploton et al. (2020) show the problem in forest biomass mapping. A standard non-spatial check suggested that a random forest model explained more than half of biomass variation, while spatial validation revealed quasi-null predictive power. Meyer and Pebesma (2021) add that cross-validation error holds only inside the area where the model has seen similar predictor values, which they call the area of applicability. For this project, held-out spatial blocks are the right unit. A crown-level F1 from random crowns would be too optimistic.

**Block bootstrap.** The block bootstrap resamples blocks of consecutive observations, not single observations, so that dependence inside a block is kept. Kunsch (1989) develops this for stationary sequences by selecting blocks randomly with replacement. Lahiri (2003) is the standard book on resampling for dependent data. Kunsch's setting is a one-dimensional sequence. Using spatial blocks of map cells or crowns is an adaptation of the same idea. The project describes it that way and uses it only to put confidence intervals on block-level scores.

## 5. Soil salinity and planting monitoring in drylands (Aral add-on)

Metternicht and Zinck (2003) review remote sensing of soil salinity. They conclude that optical and microwave time series can detect changes in salt-related surface features. They list constraints: the spectral behaviour of different salt types, how salts are spread over the surface, change over time, and interference from vegetation. The last point matters for planting monitoring, because plants and salt crusts sit in the same pixels.

Khan et al. (2005) tested simple remote sensing indicators of hydrosaline land degradation on irrigated land near Faisalabad, Pakistan. They used salinity indices built mostly from band combinations, plus vegetation indices, and found that classification accuracy depended strongly on season. Allbed and Kumar (2013) review sensors and methods for salinity mapping in arid and semi-arid regions. They cover both vegetation and salinity indices, and they highlight the main issues that limit the use of remote sensing for this task. Wang et al. (2020) compared Landsat-8 OLI and Sentinel-2 MSI for soil salinity at the Ebinur Lake Wetland in north-west China, using 64 topsoil samples. Per the published summary, the finer-resolution Sentinel-2 imagery performed better.

For dryland woody cover monitoring, Brandt et al. (2018) used very high resolution satellite imagery to map woody cover in West African drylands. They found that in the semi-arid Sahel, farmland management promoted woody cover around villages (11% cover), while sub-humid farmlands had much lower woody cover (21%) than savannas (33%). This supports using fine imagery to judge planted or managed woody cover, and then asking what 10 m data can add.

**The Aral seabed.** Salmukhanbetova et al. (2021) surveyed 24 saxaul plantations on the Kazakh part of the dried seabed in September 2020. They give a GPS point, planting year, survival, density and mean crown area for each plot; survival ranged from under 1 % to 78 %, and mean crowns were 0.2 to 7.5 m2, far smaller than a Sentinel-2 pixel. Their plot table is the reference for the add-on. Duan et al. (2022) related Landsat salinity indices to topsoil salt content on the exposed seabed and found one index worked best; its formula is in the full text, which could not be read for this review. Löw et al. (2022) mapped land cover change across the Aralkum with Landsat-8, Sentinel-2 and Sentinel-1, and estimated that about a third of the Aralkum was saline soil with high erosion risk in 2020 (abstract). Khamzina et al. (2020) studied an age sequence of *Haloxylon* plantations on sandy seabed and found higher water-use efficiency in older stands (abstract).

For the saxaul add-on, these papers set two rules. Salinity indices and MSAVI should be read together, because salt and plants mix in the same pixels. Index values should be tested by season, because accuracy changes with season.

## 6. The gap this project fills

The literature above covers crown detection, shrub mapping on Jornada, indices on sparse cover, and spatial validation, but mostly as separate topics. In the sources reviewed here, no study compares SAM crown masks (automatic and point-prompted) against an index-threshold-plus-watershed baseline, scores both against lidar-derived crowns on held-out spatial blocks with block-bootstrap intervals, and then aggregates the same crowns to 10 m cells to compare them with Sentinel-2 MSAVI and NDVI. The studies on fractional cover report error sizes but do not state a cover level at which 10 m imagery stops detecting shrubs. This project estimates that cover threshold from crown cover per 10 m cell, so that a reader can see where Sentinel-2 works and where it does not for sparse desert shrubs.

## References

Allbed, A., & Kumar, L. (2013). Soil salinity mapping and monitoring in arid and semi-arid regions using remote sensing technology: A review. *Advances in Remote Sensing, 2*(4), 373-385. https://doi.org/10.4236/ars.2013.24040

Allred, B. W., Bestelmeyer, B. T., Boyd, C. S., et al. (2021). Improving Landsat predictions of rangeland fractional cover with multitask learning and uncertainty. *Methods in Ecology and Evolution, 12*(5), 841-849. https://doi.org/10.1111/2041-210X.13564

Brandt, M., Rasmussen, K., Hiernaux, P., et al. (2018). Reduction of tree cover in West African woodlands and promotion in semi-arid farmlands. *Nature Geoscience, 11*(5), 328-333. https://doi.org/10.1038/s41561-018-0092-x

Brandt, M., Tucker, C. J., Kariryaa, A., et al. (2020). An unexpectedly large count of trees in the West African Sahara and Sahel. *Nature, 587*(7832), 78-82. https://doi.org/10.1038/s41586-020-2824-5

Duan, Z., Wang, X., & Sun, L. (2022). Monitoring and mapping of soil salinity on the exposed seabed of the Aral Sea, Central Asia. *Water, 14*(9), 1438. https://doi.org/10.3390/w14091438

Drusch, M., Del Bello, U., Carlier, S., et al. (2012). Sentinel-2: ESA's optical high-resolution mission for GMES operational services. *Remote Sensing of Environment, 120*, 25-36. https://doi.org/10.1016/j.rse.2011.11.026

Elmore, A. J., Mustard, J. F., Manning, S. J., & Lobell, D. B. (2000). Quantifying vegetation change in semiarid environments: Precision and accuracy of spectral mixture analysis and the normalized difference vegetation index. *Remote Sensing of Environment, 73*(1), 87-102. https://doi.org/10.1016/S0034-4257(00)00100-0

Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A. (2010). The Pascal Visual Object Classes (VOC) challenge. *International Journal of Computer Vision, 88*(2), 303-338. https://doi.org/10.1007/s11263-009-0275-4

Guirado, E., Blanco-Sacristan, J., Rodriguez-Caballero, E., Tabik, S., et al. (2021). Mask R-CNN and OBIA fusion improves the segmentation of scattered vegetation in very high-resolution optical sensors. *Sensors, 21*(1), 320. https://doi.org/10.3390/s21010320

He, K., Gkioxari, G., Dollar, P., & Girshick, R. (2017). Mask R-CNN. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2980-2988. https://doi.org/10.1109/ICCV.2017.322

Huete, A. R. (1988). A soil-adjusted vegetation index (SAVI). *Remote Sensing of Environment, 25*(3), 295-309. https://doi.org/10.1016/0034-4257(88)90106-X

Ji, W., Hanan, N. P., Browning, D. M., Monger, H. C., Peters, D. P. C., et al. (2019). Constraints on shrub cover and shrub-shrub competition in a U.S. southwest desert. *Ecosphere, 10*(2), e02590. https://doi.org/10.1002/ecs2.2590

Jones, M. O., Allred, B. W., Naugle, D. E., Maestas, J. D., et al. (2018). Innovation in rangeland monitoring: Annual, 30 m, plant functional type percent cover maps for U.S. rangelands, 1984-2017. *Ecosphere, 9*(9), e02430. https://doi.org/10.1002/ecs2.2430

Khamzina, A., An, J., Chang, H., & Son, Y. (2020). Isotopic signatures as an indicator of long-term water-use efficiency of *Haloxylon* plantations on the dried Aral Sea bed. *Water, 12*(1), 99. https://doi.org/10.3390/w12010099

Karlson, M., Reese, H., & Ostwald, M. (2014). Tree crown mapping in managed woodlands (parklands) of semi-arid West Africa using WorldView-2 imagery and geographic object based image analysis. *Sensors, 14*(12), 22643-22669. https://doi.org/10.3390/s141222643

Khan, N. M., Rastoskuev, V. V., Sato, Y., & Shiozawa, S. (2005). Assessment of hydrosaline land degradation by using a simple approach of remote sensing indicators. *Agricultural Water Management, 77*(1-3), 96-109. https://doi.org/10.1016/j.agwat.2004.09.038

Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., et al. (2023). Segment Anything. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 3992-4003. https://doi.org/10.1109/ICCV51070.2023.00371 (preprint: https://doi.org/10.48550/arXiv.2304.02643)

Kunsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *The Annals of Statistics, 17*(3), 1217-1241. https://doi.org/10.1214/aos/1176347265

Lahiri, S. N. (2003). *Resampling methods for dependent data*. Springer. https://doi.org/10.1007/978-1-4757-3803-2

Laliberte, A. S., & Rango, A. (2009). Texture and scale in object-based analysis of subdecimeter resolution unmanned aerial vehicle (UAV) imagery. *IEEE Transactions on Geoscience and Remote Sensing, 47*(3), 761-770. https://doi.org/10.1109/TGRS.2008.2009355

Laliberte, A. S., Goforth, M. A., Steele, C. M., & Rango, A. (2011). Multispectral remote sensing from unmanned aircraft: Image processing workflows and applications for rangeland environments. *Remote Sensing, 3*(11), 2529-2551. https://doi.org/10.3390/rs3112529

Laliberte, A. S., Rango, A., Havstad, K. M., Paris, J. F., Beck, R. F., et al. (2004). Object-oriented image analysis for mapping shrub encroachment from 1937 to 2003 in southern New Mexico. *Remote Sensing of Environment, 93*(1-2), 198-210. https://doi.org/10.1016/j.rse.2004.07.011

Löw, F., Dimov, D., Kenjabaev, S., Zaitov, S., Stulina, G., & Dukhovny, V. (2022). Land cover change detection in the Aralkum with multi-source satellite datasets. *GIScience & Remote Sensing, 59*(1), 17-35. https://doi.org/10.1080/15481603.2021.2009232

Meyer, H., & Pebesma, E. (2021). Predicting into unknown space? Estimating the area of applicability of spatial prediction models. *Methods in Ecology and Evolution, 12*(9), 1620-1633. https://doi.org/10.1111/2041-210X.13650

Metternicht, G. I., & Zinck, J. A. (2003). Remote sensing of soil salinity: Potentials and constraints. *Remote Sensing of Environment, 85*(1), 1-20. https://doi.org/10.1016/S0034-4257(02)00188-8

Mitchell, J. J., Glenn, N. F., Sankey, T. T., Derryberry, D. R., et al. (2011). Small-footprint lidar estimations of sagebrush canopy characteristics. *Photogrammetric Engineering & Remote Sensing, 77*(5), 521-530. https://doi.org/10.14358/pers.77.5.521

Okin, G. S., Roberts, D. A., Murray, B., & Okin, W. J. (2001). Practical limits on hyperspectral vegetation discrimination in arid and semiarid environments. *Remote Sensing of Environment, 77*(2), 212-225. https://doi.org/10.1016/S0034-4257(01)00207-3

Osco, L. P., Wu, Q., de Lemos, E. L., Goncalves, W. N., et al. (2023). The Segment Anything Model (SAM) for remote sensing applications: From zero to one shot. *International Journal of Applied Earth Observation and Geoinformation, 124*, 103540. https://doi.org/10.1016/j.jag.2023.103540

Ploton, P., Mortier, F., Rejou-Mechain, M., Barbier, N., Picard, N., et al. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. *Nature Communications, 11*, 4540. https://doi.org/10.1038/s41467-020-18321-y

Qi, J., Chehbouni, A., Huete, A. R., Kerr, Y. H., & Sorooshian, S. (1994). A modified soil adjusted vegetation index. *Remote Sensing of Environment, 48*(2), 119-126. https://doi.org/10.1016/0034-4257(94)90134-1

Rango, A., Chopping, M., Ritchie, J., Havstad, K., Kustas, W., & Schmugge, T. (2000). Morphological characteristics of shrub coppice dunes in desert grasslands of southern New Mexico derived from scanning LIDAR. *Remote Sensing of Environment, 74*(1), 26-44. https://doi.org/10.1016/S0034-4257(00)00084-5

Rango, A., Laliberte, A., Herrick, J. E., Winters, C., Havstad, K., et al. (2009). Unmanned aerial vehicle-based remote sensing for rangeland assessment, monitoring, and management. *Journal of Applied Remote Sensing, 3*(1), 1-15. https://doi.org/10.1117/1.3216822

Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography, 40*(8), 913-929. https://doi.org/10.1111/ecog.02881

Roberts, T. H., & Hanan, N. P. (2023). *Gridded 1-hectare estimates of shrub community structure at the Jornada Basin LTER site derived from NAIP (2011) and LiDAR (2019) data* [Data set]. Environmental Data Initiative. https://doi.org/10.6073/pasta/49bec3ac2917e494c5de4b19130e703b

Rouse, J. W., Haas, R. H., Schell, J. A., & Deering, D. W. (1974). Monitoring vegetation systems in the Great Plains with ERTS. In *Proceedings of the Third Earth Resources Technology Satellite-1 Symposium* (NASA SP-351, Vol. 1, pp. 309-317). NASA. (no DOI)

Salmukhanbetova, Zh. K., Imanalinova, A. A., Dimeyeva, L. A., & Zverev, N. E. (2021). State of saxaul plantations in Kazakhstan's section of the Aral Sea region. *Central Asian Journal of Water Research, 7*(1), 128-157. https://doi.org/10.29258/CAJWR/2020-R1.v7-1/128-157.eng

Smith, M. O., Ustin, S. L., Adams, J. B., & Gillespie, A. R. (1990). Vegetation in deserts: I. A regional measure of abundance from multispectral images. *Remote Sensing of Environment, 31*(1), 1-26. https://doi.org/10.1016/0034-4257(90)90074-V

Tucker, C. J. (1979). Red and photographic infrared linear combinations for monitoring vegetation. *Remote Sensing of Environment, 8*(2), 127-150. https://doi.org/10.1016/0034-4257(79)90013-0

Wang, J., Ding, J., Yu, D., Teng, D., He, B., et al. (2020). Machine learning-based detection of soil salinity in an arid desert region, Northwest China: A comparison between Landsat-8 OLI and Sentinel-2 MSI. *Science of the Total Environment, 707*, 136092. https://doi.org/10.1016/j.scitotenv.2019.136092

Weinstein, B. G., Marconi, S., Aubry-Kientz, M., Vincent, G., Senyondo, H., et al. (2020a). DeepForest: A Python package for RGB deep learning tree crown delineation. *Methods in Ecology and Evolution, 11*(12), 1743-1751. https://doi.org/10.1111/2041-210X.13472

Weinstein, B. G., Marconi, S., Bohlman, S. A., Zare, A., & White, E. P. (2020b). Cross-site learning in deep learning RGB tree crown detection. *Ecological Informatics, 56*, 101061. https://doi.org/10.1016/j.ecoinf.2020.101061

Weinstein, B. G., Marconi, S., Bohlman, S., Zare, A., & White, E. (2019). Individual tree-crown detection in RGB imagery using semi-supervised deep learning neural networks. *Remote Sensing, 11*(11), 1309. https://doi.org/10.3390/rs11111309

Wu, Q., & Osco, L. P. (2023). samgeo: A Python package for segmenting geospatial data with the Segment Anything Model (SAM). *Journal of Open Source Software, 8*(89), 5663. https://doi.org/10.21105/joss.05663

## Verification table

DOIs were queried at `https://api.crossref.org/works/<DOI>` (and `https://api.datacite.org/dois/<DOI>` for the data set and the arXiv preprint). "Yes" means title, first author, year and journal matched the citation above.

| Citation | DOI | Verified |
|---|---|---|
| Allbed & Kumar (2013) | 10.4236/ars.2013.24040 | yes (Crossref) |
| Allred et al. (2021) | 10.1111/2041-210X.13564 | yes (Crossref) |
| Brandt et al. (2018) | 10.1038/s41561-018-0092-x | yes (Crossref) |
| Brandt et al. (2020) | 10.1038/s41586-020-2824-5 | yes (Crossref) |
| Drusch et al. (2012) | 10.1016/j.rse.2011.11.026 | yes (Crossref) |
| Duan et al. (2022) | 10.3390/w14091438 | yes (Crossref) |
| Elmore et al. (2000) | 10.1016/S0034-4257(00)00100-0 | yes (Crossref) |
| Everingham et al. (2010) | 10.1007/s11263-009-0275-4 | yes (Crossref) |
| Guirado et al. (2021) | 10.3390/s21010320 | yes (Crossref) |
| He et al. (2017) | 10.1109/ICCV.2017.322 | yes (Crossref) |
| Huete (1988) | 10.1016/0034-4257(88)90106-X | yes (Crossref) |
| Ji et al. (2019) | 10.1002/ecs2.2590 | yes (Crossref) |
| Jones et al. (2018) | 10.1002/ecs2.2430 | yes (Crossref) |
| Khamzina et al. (2020) | 10.3390/w12010099 | yes (Crossref) |
| Karlson et al. (2014) | 10.3390/s141222643 | yes (Crossref) |
| Khan et al. (2005) | 10.1016/j.agwat.2004.09.038 | yes (Crossref) |
| Kirillov et al. (2023), ICCV | 10.1109/ICCV51070.2023.00371 | yes (Crossref) |
| Kirillov et al. (2023), arXiv | 10.48550/arXiv.2304.02643 | yes (DataCite) |
| Kunsch (1989) | 10.1214/aos/1176347265 | yes (Crossref) |
| Lahiri (2003) | 10.1007/978-1-4757-3803-2 | yes (Crossref) |
| Laliberte & Rango (2009) | 10.1109/TGRS.2008.2009355 | yes (Crossref) |
| Laliberte et al. (2011) | 10.3390/rs3112529 | yes (Crossref) |
| Laliberte et al. (2004) | 10.1016/j.rse.2004.07.011 | yes (Crossref) |
| Löw et al. (2022) | 10.1080/15481603.2021.2009232 | yes (Crossref) |
| Meyer & Pebesma (2021) | 10.1111/2041-210X.13650 | yes (Crossref) |
| Metternicht & Zinck (2003) | 10.1016/S0034-4257(02)00188-8 | yes (Crossref) |
| Mitchell et al. (2011) | 10.14358/pers.77.5.521 | yes (Crossref) |
| Okin et al. (2001) | 10.1016/S0034-4257(01)00207-3 | yes (Crossref) |
| Osco et al. (2023) | 10.1016/j.jag.2023.103540 | yes (Crossref) |
| Ploton et al. (2020) | 10.1038/s41467-020-18321-y | yes (Crossref) |
| Qi et al. (1994) | 10.1016/0034-4257(94)90134-1 | yes (Crossref) |
| Rango et al. (2000) | 10.1016/S0034-4257(00)00084-5 | yes (Crossref) |
| Rango et al. (2009) | 10.1117/1.3216822 | yes (Crossref) |
| Roberts et al. (2017) | 10.1111/ecog.02881 | yes (Crossref) |
| Roberts & Hanan (2023), data set | 10.6073/pasta/49bec3ac2917e494c5de4b19130e703b | yes (DataCite) |
| Rouse et al. (1974) | none | no DOI (NASA report in conference proceedings); standard citation given |
| Salmukhanbetova et al. (2021) | 10.29258/CAJWR/2020-R1.v7-1/128-157.eng | yes (Crossref; Crossref lists 2020, the journal issue is 2021) |
| Smith et al. (1990) | 10.1016/0034-4257(90)90074-V | yes (Crossref) |
| Tucker (1979) | 10.1016/0034-4257(79)90013-0 | yes (Crossref) |
| Wang et al. (2020) | 10.1016/j.scitotenv.2019.136092 | yes (Crossref) |
| Weinstein et al. (2020a), DeepForest | 10.1111/2041-210X.13472 | yes (Crossref) |
| Weinstein et al. (2020b) | 10.1016/j.ecoinf.2020.101061 | yes (Crossref) |
| Weinstein et al. (2019) | 10.3390/rs11111309 | yes (Crossref) |
| Wu & Osco (2023) | 10.21105/joss.05663 | yes (Crossref) |
