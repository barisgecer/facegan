function [n,xout,h] = histf(Y,varargin)
%HISTF plots formatted histograms. It's similar to hist, but with more
%options for FaceColor, BarWidth, LineStyle, FaceAlpha, etc. 
% 
%% Syntax
% 
%  histf(Y)
%  histf(Y,x)
%  histf(Y,nbins)
%  histf(...,'BarWidth',BarWidth)
%  histf(...,'FaceColor',FaceColor)
%  histf(...,'EdgeColor',EdgeColor)
%  histf(...,'LineStyle',LineStyle)
%  histf(...,'LineWidth',LineWidth)
%  histf(...,'FaceAlpha',FaceAlpha)
%  histf(...,'EdgeAlpha',EdgeAlpha)
%  histf(...,'Alpha',Alpha)
%  [n,xout,h] = histf(...)
% 
%% Description
% 
% histf(Y) bins the elements in vector Y into 10 equally spaced containers and 
% returns the number of elements in each container as a row vector. If Y is an m-by-p 
% matrix, histf treats the columns of Y as vectors. No elements of Y can be complex 
% or of type integer.
%
% histf(Y,x) where x is a vector, returns the distribution of Y among length(x)
% bins with centers specified by x. For example, if x is a 5-element vector, histf
% distributes the elements of Y into five bins centered on the x-axis at the elements in
% x, none of which can be complex.
%
% histf(Y,nbins) where nbins is a scalar, uses nbins number of bins.
%
% histf(...,'BarWidth',BarWidth) specifies the width of bars as a
% fraction of total space available for each bar. Default BarWidth is 1. 
%
% histf(...,'FaceColor',FaceColor) specifies face color as a short name, long name, or 
% RGB triple. 
%
% histf(...,'EdgeColor',EdgeColor) specifies edge coloras a short name, long name, or 
% RGB triple. 
%
% histf(...,'LineStyle',LineStyle) specifies line style of bar edges. Can
% be '-' (default), ':', '--', '-.', or 'none'.
%
% histf(...,'LineWidth',LineWidth) specifies line width of bar edge in
% points. Default LineWidth is 0.5 points. 
% 
% histf(...,'FaceAlpha',FaceAlpha) specifies transparency value of bar
% faces. FaceAlpha must be a scalar between 0 and 1. Default value is 1. 
% 
% histf(...,'EdgeAlpha',EdgeAlpha) specifies transparency value of bar
% edges. EdgeAlpha must be a scalar between 0 and 1. Default value is 1. 
% 
% histf(...,'Alpha',Alpha) sets FaceAlpha and EdgeAlpha to Alpha. 
%
% [n,xout,h] = histf(...) returns vectors n and xout containing the frequency counts 
% and the bin locations. h is the handle of the newly plotted histogram. 
% 
%% Author Info
% This function was written by Chad A. Greene of the University of Texas at
% Austin's Institute for Geophysics (UTIG). August 2014. 
% 
% See also hist, histc, bar, alpha, and legalpha.  

assert(isnumeric(Y)==1,'Input Y must be numeric.')

x_or_nbins = 10; % 10 bins by default to match hist(Y) default. 
if nargin>1 & isnumeric(varargin{1})
    x_or_nbins = varargin{1}; 
    varargin(1)=[]; 
end

FaceAlpha = 1; 
EdgeAlpha = 1; 
tmp = strcmpi(varargin,'facealpha');
if any(tmp)
    FaceAlpha = varargin{find(tmp)+1};
    assert(FaceAlpha>=0&FaceAlpha<=1,'FaceAlpha value must be between 0 and 1.')
    tmp(find(tmp)+1)=1; 
    varargin = varargin(~tmp); 
end

tmp = strcmpi(varargin,'edgealpha'); 
if any(tmp)
    EdgeAlpha = varargin{find(tmp)+1};
    assert(EdgeAlpha>=0&EdgeAlpha<=1,'EdgeAlpha value must be between 0 and 1.') 
    tmp(find(tmp)+1)=1; 
    varargin = varargin(~tmp); 
end    

% User can define simply "alpha" to set facealpha and edgealpha: 
tmp = strcmpi(varargin,'alpha'); 
if any(tmp)
    EdgeAlpha = varargin{find(tmp)+1};
    FaceAlpha = varargin{find(tmp)+1};
    assert(EdgeAlpha>=0&EdgeAlpha<=1,'Alpha value must be between 0 and 1.') 
    tmp(find(tmp)+1)=1; 
    varargin = varargin(~tmp); 
end   

BarWidth = 1; % to match the default of hist
tmp = strcmpi(varargin,'barwidth'); 
if any(tmp)
    BarWidth = varargin{find(tmp)+1};
    assert(isscalar(BarWidth)==1,'BarWidth must be a scalar value between 0 and 1.') 
    tmp(find(tmp)+1)=1; 
    varargin = varargin(~tmp); 
end    

[n,xout] = hist(Y,x_or_nbins);
h = bar(xout,n,'BarWidth',BarWidth,varargin{:}); 
h.FaceAlpha = FaceAlpha;
h.EdgeAlpha = EdgeAlpha;
halpha = findobj(h,'Type','patch');
set(halpha,'FaceAlpha',FaceAlpha,'EdgeAlpha',EdgeAlpha); 

if nargout==0
    clear n xout h 
end

end