package plugins.adufour.hierarchicalkmeans;

import java.awt.Point;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.swing.JSeparator;

import icy.image.IcyBufferedImage;
import icy.image.colormap.FireColorMap;
import icy.main.Icy;
import icy.roi.ROI;
import icy.roi.ROI2D;
import icy.roi.ROI3D;
import icy.sequence.Sequence;
import icy.sequence.SequenceDataIterator;
import icy.sequence.SequenceUtil;
import icy.swimmingPool.SwimmingObject;
import icy.type.DataIteratorUtil;
import icy.type.DataType;
import icy.type.point.Point5D;
import icy.util.OMEUtil;
import loci.formats.ome.OMEXMLMetadataImpl;
import plugins.adufour.blocks.lang.Block;
import plugins.adufour.blocks.util.VarList;
import plugins.adufour.connectedcomponents.ConnectedComponent;
import plugins.adufour.ezplug.EzLabel;
import plugins.adufour.ezplug.EzPlug;
import plugins.adufour.ezplug.EzStoppable;
import plugins.adufour.ezplug.EzVar;
import plugins.adufour.ezplug.EzVarBoolean;
import plugins.adufour.ezplug.EzVarChannel;
import plugins.adufour.ezplug.EzVarDouble;
import plugins.adufour.ezplug.EzVarFrame;
import plugins.adufour.ezplug.EzVarInteger;
import plugins.adufour.ezplug.EzVarListener;
import plugins.adufour.ezplug.EzVarSequence;
import plugins.adufour.filtering.ConvolutionException;
import plugins.adufour.vars.lang.VarROIArray;
import plugins.adufour.vars.lang.VarSequence;
import plugins.adufour.vars.util.VarException;
import plugins.kernel.roi.descriptor.measure.ROIMassCenterDescriptorsPlugin;
import plugins.nchenouard.spot.DetectionResult;
import plugins.nchenouard.spot.Point3D;
import plugins.nchenouard.spot.Spot;

public class HierarchicalKMeans extends EzPlug implements Block, EzStoppable
{
    protected static int resultID = 1;
    
    protected EzVarSequence input = new EzVarSequence("Input");
    
    protected EzVarChannel channel = new EzVarChannel("Channel", input.getVariable(), true);
    
    protected EzVarFrame frame = new EzVarFrame("Frame", input.getVariable(), true);
    
    protected EzVarDouble preFilterSigma = new EzVarDouble("Gaussian pre-filter", 0, 50, 0.1);
    
    protected EzVarInteger minSize = new EzVarInteger("Min object size (px)", 100, 1, 200000000, 1);
    
    protected EzVarInteger maxSize = new EzVarInteger("Max object size (px)", 1600, 1, 200000000, 1);
    
    protected EzVarInteger nbClasses = new EzVarInteger("Intensity classes", 10, 2, 255, 1);
    
    protected EzVarDouble finalThreshold = new EzVarDouble("Min object intensity", 0, 0, 65535, 1);
    
    protected EzVarBoolean exportROI      = new EzVarBoolean("Export ROIs", true);
    protected EzVarBoolean exportSequence = new EzVarBoolean("Export labels", false);
    protected EzVarBoolean exportSwPool   = new EzVarBoolean("Prepare for tracking", false);
    
    protected EzLabel nbObjects = new EzLabel(" ");
    
    protected VarSequence outputSequence = new VarSequence("binary sequence", null);
    
    protected VarROIArray outputROIs = new VarROIArray("list of ROI");
    
    @Override
    public void initialize()
    {
        addEzComponent(input);
        
        input.addVarChangeListener(new EzVarListener<Sequence>()
        {
            @Override
            public void variableChanged(EzVar<Sequence> source, Sequence newValue)
            {
                exportSwPool.setVisible(newValue != null && newValue.getSizeT() > 1);
            }
        });
        
        addEzComponent(frame);
        addEzComponent(channel);
        addEzComponent(preFilterSigma);
        
        addComponent(new JSeparator(JSeparator.HORIZONTAL));
        
        // Number of classes
        String nbClassesHelp = "<html>A classical threshold splits the histogram into 2 classes (background and foreground)<br/>";
        nbClassesHelp += "Increase this value if the objects of interest have different mean intensities</html>";
        nbClasses.setToolTipText(nbClassesHelp);
        addEzComponent(nbClasses);
        
        // Size constraint
        String minSizeHelp = "<html>Objects with an area (2D) or volume (3D) below this value are discarded</html>";
        String maxSizeHelp = "<html>Objects with an area (2D) or volume (3D) above this value are discarded</html>";
        minSize.setToolTipText(minSizeHelp);
        maxSize.setToolTipText(maxSizeHelp);
        addEzComponent(minSize);
        addEzComponent(maxSize);
        
        // Final threshold
        String finalThresholdHelp = "<html>Objects are considerered valid if they have at least one pixel above this value<br/>";
        finalThresholdHelp += "=> useful to remove spurious artefacts from the background</html>";
        finalThreshold.setToolTipText(finalThresholdHelp);
        addEzComponent(finalThreshold);
        
        addComponent(new JSeparator(JSeparator.HORIZONTAL));
        
        addEzComponent(exportROI);
        addEzComponent(exportSequence);
        addEzComponent(exportSwPool);
        exportSwPool.setToolTipText("Exports the detected object in a format compatible with the \"Spot Tracking\" plug-in");
        
        addComponent(new JSeparator(JSeparator.HORIZONTAL));
        
        addEzComponent(nbObjects);
    }
    
    @Override
    public void execute()
    {
        Sequence _inSeq = input.getValue(true);
        Sequence _outSeq = null;
        
        int minT = frame.getValue(), sizeT = 1;
        if (minT == -1)
        {
            minT = 0;
            sizeT = _inSeq.getSizeT();
        }
        
        int minC = channel.getValue(), sizeC = minC;
        if (minC == -1)
        {
            minC = 0;
            sizeC = _inSeq.getSizeC();
        }
        
        if (exportSequence.getValue() || outputSequence.isReferenced())
        {
            // initialize the output sequence
            
            OMEXMLMetadataImpl metadata = OMEUtil.createOMEMetadata(_inSeq.getMetadata());
            String name = _inSeq.getName() + "_HK-Means" + (isHeadLess() ? "" : ("#" + resultID++));
            _outSeq = new Sequence(metadata, name);
            
            if (sizeT > 1)
            {
                for (int t = minT; t < minT + sizeT; t++)
                    for (int z = 0; z < _inSeq.getSizeZ(); z++)
                        _outSeq.setImage(t, z, new IcyBufferedImage(_inSeq.getWidth(), _inSeq.getHeight(), sizeC, DataType.USHORT));
            }
            else
            {
                for (int z = 0; z < _inSeq.getSizeZ(); z++)
                    _outSeq.setImage(0, z, new IcyBufferedImage(_inSeq.getWidth(), _inSeq.getHeight(), sizeC, DataType.USHORT));
            }
            
            outputSequence.setValue(_outSeq);
        }
        
        byte nbKMeansClasses = nbClasses.getValue().byteValue();
        if (nbKMeansClasses < 2) throw new VarException(nbClasses.getVariable(), "HK-Means requires at least two classes to run");
        
        List<ROI> detections = HKMeans.hKMeans(_inSeq, frame.getValue(), channel.getValue(), preFilterSigma.getValue(), nbKMeansClasses, minSize.getValue(), maxSize.getValue(),
                finalThreshold.getValue(), getStatus());
                
        // Rename and store the detections
        int detectionID = 1;
        for (ROI detection : detections)
            detection.setName("HK-Means detection #" + detectionID++);
        outputROIs.setValue(detections.toArray(new ROI[detections.size()]));
        
        if (exportROI.getValue())
        {
            for (ROI roi : input.getValue().getROIs())
                if (roi.getName().startsWith("HK-Means")) _inSeq.removeROI(roi, false);
                
            for (ROI roi : outputROIs.getValue())
                _inSeq.addROI(roi, false);
        }
        
        if (getUI() != null) nbObjects.setText(detections.size() + " objects detected");
        
        if (_outSeq != null)
        {
            // Generate the labels from the extracted ROI
            
            int roiID = 1;
            for (ROI roi : detections)
                DataIteratorUtil.set(new SequenceDataIterator(_outSeq, roi), roiID++);
                
            _outSeq.updateChannelsBounds(true);
            
            if (channel.getValue() == -1)
            {
                // Use same color maps as the original sequence
                for (int c = 0; c < input.getValue().getSizeC(); c++)
                    _outSeq.setColormap(c, _inSeq.getColorMap(c), true);
            }
            else
            {
                // Use a "fire" color map
                _outSeq.getColorModel().setColorMap(0, new FireColorMap(), true);
            }
            
            if (!isHeadLess()) addSequence(_outSeq);
        }
        
        if (exportSwPool.getValue())
        {
            // Convert the list of ROI to a detection set
            DetectionResult result = new DetectionResult();
            result.setSequence(_inSeq);
            for (ROI roi : detections)
            {
                Point5D center = ROIMassCenterDescriptorsPlugin.computeMassCenter(roi);
                Spot trackableSpot = new Spot(center.getX(), center.getY(), center.getZ());
                
                if (roi instanceof ROI2D)
                {
                    ROI2D r2 = (ROI2D) roi;
                    
                    for (Point pt : r2.getBooleanMask(true).getPoints())
                        trackableSpot.point3DList.add(new Point3D(pt.x, pt.y, r2.getZ()));
                        
                    result.addDetection(r2.getT(), trackableSpot);
                }
                else if (roi instanceof ROI3D)
                {
                    ROI3D r3 = (ROI3D) roi;
                    
                    for (icy.type.point.Point3D.Integer pt : r3.getBooleanMask(true).getPoints())
                        trackableSpot.point3DList.add(new Point3D(pt.x, pt.y, pt.z));
                        
                    result.addDetection(r3.getT(), trackableSpot);
                }
            }
            SwimmingObject object = new SwimmingObject(result, "HK-Means: " + result.getNumberOfDetection() + " objects");
            Icy.getMainInterface().getSwimmingPool().add(object);
        }
    }
    
    /**
     * Performs a hierarchical K-Means segmentation on the input sequence, and returns the result as
     * a labeled sequence
     * 
     * @deprecated Use {@link HKMeans#hKMeans(Sequence, double, int, int, int, Double, Sequence)}
     *             instead.
     * @param seqIN
     *            the sequence to segment
     * @param preFilter
     *            the standard deviation of the Gaussian filter to apply before segmentation (0 for
     *            none)
     * @param nbKMeansClasses
     *            the number of classes to divide the histogram
     * @param minSize
     *            the minimum size in pixels of the objects to segment
     * @param maxSize
     *            the maximum size in pixels of the objects to segment
     * @return a labeled sequence with all objects extracted in the different classes
     * @throws ConvolutionException
     *             if the filter size is too large w.r.t. the image size
     */
    public static Sequence hierarchicalKMeans(Sequence seqIN, double preFilter, int nbKMeansClasses, int minSize, int maxSize) throws ConvolutionException
    {
        return hierarchicalKMeans(seqIN, preFilter, nbKMeansClasses, minSize, maxSize, (Double) null);
    }
    
    /**
     * Performs a hierarchical K-Means segmentation on the input sequence, and returns the result as
     * a labeled sequence
     * 
     * @deprecated Use {@link HKMeans#hKMeans(Sequence, double, int, int, int, Double, Sequence)}
     *             instead.
     * @param seqIN
     *            the sequence to segment
     * @param preFilter
     *            the standard deviation of the Gaussian filter to apply before segmentation (0 for
     *            none)
     * @param nbKMeansClasses
     *            the number of classes to divide the histogram
     * @param minSize
     *            the minimum size in pixels of the objects to segment
     * @param maxSize
     *            the maximum size in pixels of the objects to segment
     * @param minValue
     *            the minimum intensity value each object should have (in any of the input channels)
     * @return a labeled sequence with all objects extracted in the different classes
     * @throws ConvolutionException
     *             if the filter size is too large w.r.t. the image size
     */
    public static Sequence hierarchicalKMeans(Sequence seqIN, double preFilter, int nbKMeansClasses, int minSize, int maxSize, Double minValue) throws ConvolutionException
    {
        Sequence result = new Sequence();
        
        HKMeans.hKMeans(seqIN, preFilter, nbKMeansClasses, minSize, maxSize, minValue, result);
        
        return result;
    }
    
    /**
     * Performs a hierarchical K-Means segmentation on the input sequence, and returns all the
     * detected objects
     * 
     * @deprecated Use {@link HKMeans#hKMeans(Sequence, byte, int, int, Double)} instead.
     * @param seqIN
     *            the sequence to segment
     * @param preFilter
     *            the standard deviation of the Gaussian filter to apply before segmentation (0 for
     *            none)
     * @param nbKMeansClasses
     *            the number of classes to divide the histogram
     * @param minSize
     *            the minimum size in pixels of the objects to segment
     * @param maxSize
     *            the maximum size in pixels of the objects to segment
     * @param seqOUT
     *            an empty sequence that will receive the labeled output as unsigned short, or null
     *            if not necessary
     * @return a map containing the list of connected components found in each time point
     * @throws ConvolutionException
     *             if the filter size is too large w.r.t. the image size
     */
    public static Map<Integer, List<ConnectedComponent>> hierarchicalKMeans(Sequence seqIN, double preFilter, int nbKMeansClasses, int minSize, int maxSize, Sequence seqOUT)
            throws ConvolutionException
    {
        return hierarchicalKMeans(seqIN, preFilter, nbKMeansClasses, minSize, maxSize, null, seqOUT);
    }
    
    /**
     * Performs a hierarchical K-Means segmentation on the input sequence, and returns all the
     * detected objects
     * 
     * @deprecated Use {@link HKMeans#hKMeans(Sequence, byte, int, int, Double)} instead.
     * @param seqIN
     *            the sequence to segment
     * @param preFilter
     *            the standard deviation of the Gaussian filter to apply before segmentation (0 for
     *            none)
     * @param nbKMeansClasses
     *            the number of classes to divide the histogram
     * @param minSize
     *            the minimum size in pixels of the objects to segment
     * @param maxSize
     *            the maximum size in pixels of the objects to segment
     * @param minValue
     *            the minimum intensity value each object should have (in its corresponding channel)
     * @param seqOUT
     *            an empty sequence that will receive the labeled output as unsigned short, or null
     *            if not necessary
     * @return a map containing the list of connected components found in each time point
     * @throws ConvolutionException
     *             if the filter size is too large w.r.t the image size
     */
    public static Map<Integer, List<ConnectedComponent>> hierarchicalKMeans(Sequence seqIN, double preFilter, int nbKMeansClasses, int minSize, int maxSize, Double minValue,
            Sequence seqOUT) throws ConvolutionException
    {
        return hierarchicalKMeans(seqIN, -1, preFilter, nbKMeansClasses, minSize, maxSize, minValue, seqOUT);
    }
    
    /**
     * Performs a hierarchical K-Means segmentation on the input sequence, and returns all the
     * detected objects
     * 
     * @deprecated Use {@link HKMeans#hKMeans(Sequence, byte, int, int, Double)} instead.
     * @param seqIN
     *            the sequence to segment
     * @param channel
     *            the channel to process (or -1 to process all channels)
     * @param preFilter
     *            the standard deviation of the Gaussian filter to apply before segmentation (0 for
     *            none)
     * @param nbKMeansClasses
     *            the number of classes to divide the histogram
     * @param minSize
     *            the minimum size in pixels of the objects to segment
     * @param maxSize
     *            the maximum size in pixels of the objects to segment
     * @param minValue
     *            the minimum intensity value each object should have (in its corresponding channel)
     * @param seqOUT
     *            an empty sequence that will receive the labeled output as unsigned short, or null
     *            if not necessary
     * @return a map containing the list of connected components found in each time point
     * @throws ConvolutionException
     *             if the filter size is too large w.r.t the image size
     */
    public static Map<Integer, List<ConnectedComponent>> hierarchicalKMeans(Sequence seqIN, int channel, double preFilter, int nbKMeansClasses, int minSize, int maxSize,
            Double minValue, Sequence seqOUT) throws ConvolutionException
    {
        Map<Integer, List<ConnectedComponent>> map = new HashMap<Integer, List<ConnectedComponent>>();
        
        if (seqIN.getSizeC() > 1 && channel != -1)
        {
            // extract channel
            seqIN = SequenceUtil.extractChannel(seqIN, channel);
        }
        
        List<ConnectedComponent> components = HKMeans.hKMeans(seqIN, preFilter, nbKMeansClasses, minSize, maxSize, minValue, seqOUT);
        
        // sort components by time
        for (int t = 0; t < seqIN.getSizeT(); t++)
        {
            ArrayList<ConnectedComponent> listT = new ArrayList<ConnectedComponent>();
            
            for (ConnectedComponent cc : components)
                if (cc.getT() == t) listT.add(cc);
                
            listT.trimToSize();
            map.put(t, listT);
        }
        
        return map;
    }
    
    public void clean()
    {
    }
    
    @Override
    public void declareInput(VarList inputMap)
    {
        inputMap.add("Input", input.getVariable());
        inputMap.add("Gaussian pre-filter", preFilterSigma.getVariable());
        inputMap.add("Frame", frame.getVariable());
        inputMap.add("Number of classes", nbClasses.getVariable());
        inputMap.add("Min size (px)", minSize.getVariable());
        inputMap.add("Max size (px)", maxSize.getVariable());
        inputMap.add("Final threshold", finalThreshold.getVariable());
        
        // force sequence export in box mode
        exportROI.setValue(false);
        exportSwPool.setValue(false);
        exportSequence.setValue(false);
    }
    
    @Override
    public void declareOutput(VarList outputMap)
    {
        outputMap.add("binary sequence", outputSequence);
        outputMap.add("output regions", outputROIs);
    }
    
}
