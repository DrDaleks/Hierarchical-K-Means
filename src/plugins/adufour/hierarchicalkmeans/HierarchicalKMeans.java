package plugins.adufour.hierarchicalkmeans;

import icy.image.colormap.FireColorMap;
import icy.image.colormodel.IcyColorModel;
import icy.main.Icy;
import icy.roi.ROI;
import icy.sequence.DimensionId;
import icy.sequence.Sequence;
import icy.sequence.SequenceUtil;
import icy.swimmingPool.SwimmingObject;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.vecmath.Point3i;

import plugins.adufour.blocks.lang.Block;
import plugins.adufour.blocks.util.VarList;
import plugins.adufour.connectedcomponents.ConnectedComponent;
import plugins.adufour.connectedcomponents.ConnectedComponents;
import plugins.adufour.connectedcomponents.ConnectedComponents.Sorting;
import plugins.adufour.ezplug.EzException;
import plugins.adufour.ezplug.EzGroup;
import plugins.adufour.ezplug.EzLabel;
import plugins.adufour.ezplug.EzPlug;
import plugins.adufour.ezplug.EzStoppable;
import plugins.adufour.ezplug.EzVarBoolean;
import plugins.adufour.ezplug.EzVarChannel;
import plugins.adufour.ezplug.EzVarDimensionPicker;
import plugins.adufour.ezplug.EzVarDouble;
import plugins.adufour.ezplug.EzVarEnum;
import plugins.adufour.ezplug.EzVarInteger;
import plugins.adufour.ezplug.EzVarSequence;
import plugins.adufour.filtering.ConvolutionException;
import plugins.adufour.vars.lang.VarGenericArray;
import plugins.adufour.vars.lang.VarROIArray;
import plugins.adufour.vars.lang.VarSequence;
import plugins.adufour.vars.util.VarException;
import plugins.kernel.roi.roi2d.ROI2DArea;
import plugins.kernel.roi.roi3d.ROI3DArea;
import plugins.nchenouard.spot.DetectionResult;
import plugins.nchenouard.spot.Point3D;
import plugins.nchenouard.spot.Spot;

public class HierarchicalKMeans extends EzPlug implements Block, EzStoppable
{
    protected static int                            resultID          = 1;
    
    protected EzVarSequence                         input             = new EzVarSequence("Input");
    
    protected EzVarChannel                          channel           = new EzVarChannel("channel", input.getVariable(), true);
    
    protected EzVarDimensionPicker                  frame             = new EzVarDimensionPicker("Frame", DimensionId.T, input.getVariable(), true);
    
    protected EzVarDouble                           preFilterValue    = new EzVarDouble("Gaussian pre-filter", 0, 50, 0.1);
    
    protected EzVarInteger                          minSize           = new EzVarInteger("Min size (px)", 100, 1, 200000000, 1);
    
    protected EzVarInteger                          maxSize           = new EzVarInteger("Max size (px)", 1600, 1, 200000000, 1);
    
    protected EzVarInteger                          smartLabelClasses = new EzVarInteger("Number of classes", 10, 2, 255, 1);
    
    protected EzVarDouble                           finalThreshold    = new EzVarDouble("Final threshold", 0, 0, Short.MAX_VALUE, 1);
    
    protected EzVarBoolean                          exportSequence    = new EzVarBoolean("Labeled sequence", false);
    protected EzVarBoolean                          exportSwPool      = new EzVarBoolean("Swimming pool data", false);
    protected EzVarBoolean                          exportROI         = new EzVarBoolean("ROIs", true);
    
    protected EzVarEnum<Sorting>                    sorting           = new EzVarEnum<ConnectedComponents.Sorting>("Sorting", Sorting.values(), Sorting.DEPTH_ASC);
    
    protected EzLabel                               nbObjects;
    
    protected VarSequence                           outputSequence    = new VarSequence("binary sequence", null);
    
    protected VarGenericArray<ConnectedComponent[]> outputCCs         = new VarGenericArray<ConnectedComponent[]>("objects", ConnectedComponent[].class, null);
    
    protected VarROIArray                           outputROIs        = new VarROIArray("list of ROI");
    
    @Override
    public void initialize()
    {
        addEzComponent(input);
        addEzComponent(frame);
        addEzComponent(channel);
        channel.setToolTipText("Channel to process (-1 for \"all\")");
        addEzComponent(preFilterValue);
        addEzComponent(new EzGroup("Object size", minSize, maxSize));
        addEzComponent(finalThreshold);
        addEzComponent(smartLabelClasses);
        
        addEzComponent(new EzGroup("Show result as...", exportSequence, sorting, exportSwPool, exportROI));
        exportROI.setToolTipText("Create ROIs on the original sequence");
        exportSequence.addVisibilityTriggerTo(sorting, true);
        
        addEzComponent(nbObjects = new EzLabel("< click run to start the detection >"));
    }
    
    @Override
    public void execute()
    {
        Sequence labeledSequence = null;
        
        if (exportSequence.getValue() || outputSequence.isReferenced())
        {
            labeledSequence = new Sequence(input.getValue(true).getName() + "_HK-Means" + (isHeadLess() ? "" : ("#" + resultID++)));
            outputSequence.setValue(labeledSequence);
        }
        
        int nbKMeansClasses = smartLabelClasses.getValue();
        if (nbKMeansClasses < 2) throw new VarException(smartLabelClasses.getVariable(), "HK-Means requires at least two classes to run");
        
        List<ConnectedComponent> ccs = null;
        
        try
        {
            Sequence s = input.getValue(true);
            
            // restrict to a specific channel?
            if (channel.getValue() != -1 && s.getSizeC() > 1) s = SequenceUtil.extractChannel(s, channel.getValue());
            
            // restrict to a specific frame?
            if (frame.getValue() != -1 && s.getSizeT() > 1) s = SequenceUtil.extractFrame(s, frame.getValue());
            
            // do it!
            ccs = HKMeans.hKMeans(s, preFilterValue.getValue(), smartLabelClasses.getValue(), minSize.getValue(), maxSize.getValue(), finalThreshold.getValue(), labeledSequence);
            
            // if a specific frame was extracted, the T is now incorrect, so fix it
            if (frame.getValue() != -1 && s.getSizeT() > 1)
            {
                for (ConnectedComponent cc : ccs)
                    cc.setT(frame.getValue());
            }
        }
        catch (ConvolutionException e)
        {
            throw new EzException(this, e.getMessage(), true);
        }
        
        if (getUI() != null) nbObjects.setText(ccs.size() + " objects detected");
        
        outputCCs.setValue(ccs.toArray(new ConnectedComponent[ccs.size()]));
        
        if (labeledSequence != null)
        {
            if (sorting.getValue().comparator != null) reLabel(labeledSequence, ccs, sorting.getValue().comparator);
            
            labeledSequence.updateChannelsBounds(true);
            IcyColorModel cmIN = input.getValue().getColorModel();
            IcyColorModel cmOUT = labeledSequence.getColorModel();
            
            if (channel.getValue() == -1)
            {
                // Use same color maps as the original sequence
                for (int c = 0; c < input.getValue().getSizeC(); c++)
                    cmOUT.setColorMap(c, cmIN.getColorMap(c), true);
            }
            else
            {
                // Use a "fire" color map
                labeledSequence.getColorModel().setColorMap(0, new FireColorMap(), true);
            }
            
            if (!isHeadLess()) addSequence(labeledSequence);
        }
        
        if (exportSwPool.getValue())
        {
            // Convert the list of ROI to a detection set
            DetectionResult result = new DetectionResult();
            result.setSequence(input.getValue(true));
            for (ConnectedComponent cc : ccs)
            {
                Spot trackableSpot = new Spot(cc.getX(), cc.getY(), cc.getZ());
                for (Point3i pt : cc)
                {
                    trackableSpot.point3DList.add(new Point3D(pt.x, pt.y, pt.z));
                }
                result.addDetection(cc.getT(), trackableSpot);
            }
            SwimmingObject object = new SwimmingObject(result, "HK-Means: " + result.getNumberOfDetection() + " objects");
            Icy.getMainInterface().getSwimmingPool().add(object);
        }
        
        if (exportROI.getValue() || outputROIs.isReferenced())
        {
            ROI[] rois = new ROI[ccs.size()];
            
            boolean is3D = input.getValue(true).getSizeZ() > 1;
            
            int cpt = 0;
            for (ConnectedComponent cc : ccs)
            {
                ROI roi;
                if (is3D)
                {
                    ROI3DArea area = new ROI3DArea();
                    for (Point3i pt : cc)
                        area.addPoint(pt.x, pt.y, pt.z);
                    
                    area.setT(cc.getT());
                    roi = area;
                }
                else
                {
                    ROI2DArea area = new ROI2DArea();
                    for (Point3i pt : cc)
                        area.addPoint(pt.x, pt.y);
                    area.setT(cc.getT());
                    roi = area;
                }
                
                rois[cpt++] = roi;
                roi.setName("HK-Means detection #" + cpt);
            }
            outputROIs.setValue(rois);
            
            if (exportROI.getValue())
            {
                Sequence in = input.getValue();
                
                in.beginUpdate();
                
                for (ROI roi : input.getValue().getROIs())
                    if (roi.getName().startsWith("HK-Means")) in.removeROI(roi);
                
                for (ROI roi : outputROIs.getValue())
                    in.addROI(roi);
                
                in.endUpdate();
            }
        }
    }
    
    private void reLabel(Sequence labeledSequence, List<ConnectedComponent> ccs, Comparator<ConnectedComponent> comparator)
    {
        if (comparator == null) return;
        
        int width = labeledSequence.getSizeX();
        
        // one incremental IDs for each channel
        int[] ids = new int[labeledSequence.getSizeC()];
        Arrays.fill(ids, 1);
        
        Collections.sort(ccs, comparator);
        
        for (ConnectedComponent cc : ccs)
        {
            int[][] z_xy = labeledSequence.getDataXYZAsInt(cc.getT(), cc.getC());
            int value = ids[cc.getC()];
            
            for (Point3i pt : cc)
            {
                z_xy[pt.z][pt.y * width + pt.x] = value;
            }
            
            // increment the ID for that channel
            ids[cc.getC()]++;
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
        inputMap.add("Gaussian pre-filter", preFilterValue.getVariable());
        inputMap.add("Min size (px)", minSize.getVariable());
        inputMap.add("Max size (px)", maxSize.getVariable());
        inputMap.add("Number of classes", smartLabelClasses.getVariable());
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
        outputMap.add("output objects", outputCCs);
        outputMap.add("output regions", outputROIs);
    }
    
}
