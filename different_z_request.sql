
SELECT            
SH.GalaxyID,           
SH.Redshift as z ,              
SH.SubGroupNumber,          
SH.CentreOfPotential_x, SH.CentreOfPotential_y, SH.CentreOfPotential_z,          
SH.Image_Face as face         
FROM            
RecalL0025N0752_Subhalo as SH ,            
-- Acts as a reference point            
RecalL0025N0752_Subhalo as REF            
-- Apply the conditions            
WHERE            
REF.GalaxyID=746518 and -- GalaxyID at z=0            
-- To find descendants            
((SH.SnapNum > REF.SnapNum and REF.GalaxyID            
between SH.GalaxyID and SH.TopLeafID ) or            
-- To find progenitors            
( SH.SnapNum <= REF.SnapNum and SH.GalaxyID            
between REF.GalaxyID and REF.TopLeafID ))            
-- Order the output by redshift            
ORDER BY            
SH.Redshift