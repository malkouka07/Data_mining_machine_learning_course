
SELECT TOP 5000
    s.specObjID, s.bestObjID AS objID,
    s.plate, s.mjd, s.fiberID, s.run2d,
    s.ra, s.dec, s.z, s.zErr, s.zWarning,
    p.cModelMag_g, p.cModelMag_r, p.cModelMag_i,
    p.cModelMagErr_g, p.cModelMagErr_r, p.cModelMagErr_i
FROM SpecObj AS s
JOIN PhotoObjAll AS p ON p.objID = s.bestObjID
WHERE s.class = 'GALAXY'
    AND s.zWarning = 0 AND s.z > 0
    AND p.clean = 1
    AND p.cModelMag_g BETWEEN 0 AND 40
    AND p.cModelMag_r BETWEEN 0 AND 40
    AND p.cModelMag_i BETWEEN 0 AND 40
    AND p.cModelMagErr_g > 0
    AND p.cModelMagErr_r > 0
    AND p.cModelMagErr_i > 0
ORDER BY CHECKSUM(s.specObjID, 42), s.specObjID
