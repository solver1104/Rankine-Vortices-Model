# Simulates Rankine vortices advecting each other. Optionally also renders the motion of a grid of
# tracer particles. and their trajectories over time.

# Using these packages for parallel processing
using Distributed
using SharedArrays

# Using @everywhere tags includes the packages in all processes
# Using GRUtils for plots
@everywhere using GRUtils

# Using Dates for timing when debugging/testing efficiency
@everywhere using Dates

# Using Distributions for generating random numbers
@everywhere using Distributions

# Toggle depending on whether the code is running on the Deep server or on the laptop
# Changes settings such as number of processes used and where the images are being saved
@everywhere LINUX = false

# Starts more processes
if (LINUX == true)
    addprocs(7)
else
    addprocs(3)
    end

# Number of frames to render
@everywhere framesRun = 500
# Number of vortices
@everywhere vnum = 50
# Size of the field to render
@everywhere framesz = 2
# dt in Bashforth-Adams method to solve differential equations for vortices' velocities
@everywhere deltatime = 0.1
# Toggles whether tracers should be rendered or not
@everywhere renderTracers = true
# Number of tracers per row in the initial grid of tracers
@everywhere tdensity = 20
# Toggles whether the trajectories of the tracers should be rendered or not
@everywhere renderTrajectories = true
# Length of the trajectory being renderd before it becomes invisible
@everywhere trajectoryLength = 50

# Safety measure to ensure that the trajectories of the tracers should not be rendered if the
# tracers are not being simulated and rendered
if (renderTracers == false)
    renderTrajectories = false
    end

# Use SharedArrays so that all processes can efficiently access and update arrays
# Coordinates, radii, and velocities of vortices and the virtual vortices outside the field
xz = SharedArray{Float64}(9 * vnum)
yz = SharedArray{Float64}(9 * vnum)
rz = SharedArray{Float64}(9 * vnum)
vz = SharedArray{Float64}(9 * vnum)

# Coordinates of vortices and virtual vortices during previous frame
pxz = SharedArray{Float64}(9 * vnum)
pyz = SharedArray{Float64}(9 * vnum)

# Temporary arrays used to store calculated coordinates for vortices during next frame
nxz = SharedArray{Float64}(vnum)
nyz = SharedArray{Float64}(vnum)

# Coordinates, sizes, and colors (with respect to the colormap) of the tracers
txz = SharedArray{Float64}(tdensity * tdensity)
tyz = SharedArray{Float64}(tdensity * tdensity)
tsz = SharedArray{Float64}(tdensity * tdensity)
tcz = SharedArray{Float64}(tdensity * tdensity)

# Temporary arrays used to store calculated Coordinates of the tracers during the next frame
ntxz = SharedArray{Float64}(tdensity * tdensity)
ntyz = SharedArray{Float64}(tdensity * tdensity)

# Coordinates of tracers during previous frame
ptxz = SharedArray{Float64}(tdensity * tdensity)
ptyz = SharedArray{Float64}(tdensity * tdensity)

# x and y coordinates forming a grid of positions where velocity vectors should be drawn
a = SharedArray{Float64}((20 * framesz + 1) * (20 * framesz + 1))
b = SharedArray{Float64}((20 * framesz + 1) * (20 * framesz + 1))

# x and y components of velocity vectors at every point on the grid created using a and b, defined
# above
c = SharedArray{Float64}((20 * framesz + 1) * (20 * framesz + 1))
d = SharedArray{Float64}((20 * framesz + 1) * (20 * framesz + 1))

# Set the colormap (modifies the colors of the vortices and tracers on the renders)
GRUtils.colormap("Coolwarm")

# Create and resize main Figure to plot the main plot on
mainfig = Figure((500,500))

# Array to keep track of trajectory lines to display
trajectories = []

# Randomly initialize vortices' positions, radii, and velocities
for i in 1:vnum
    xz[i] = rand(Uniform(-framesz,framesz))
    yz[i] = rand(Uniform(-framesz,framesz))
    rz[i] = 0.1
    if (rand(Uniform(-framesz/5,framesz/5)) > 0)
        vz[i] = -0.1
    else
        vz[i] = 0.1
        end

    # Initializes radii and velocities of virtual vortices (which should just be copies of the
    # radii and velocities of the vortices)
    for y in 1:8
        rz[8 * (i-1) + y + vnum] = rz[i]
        vz[8 * (i-1) + y + vnum] = vz[i]
        end
    end

# Initializes a and b
# @sync tag is used so that the processes halt once their portion of the computations is complete
# to wait for all the processes to complete
# @distributed tag is used so that the computations are distributed among the processes
@sync @distributed for i in 1:20*framesz + 1
    for j in 1:20*framesz + 1
        a[(20 * framesz + 1) * (i - 1) + j] = (i-(10 * framesz+1))/10
        b[(20 * framesz + 1) * (i - 1) + j] = (j-(10 * framesz + 1))/10
        end
    end

if (renderTracers)
    # Initializes tracer coordinates, sizes, and colors
    @sync @distributed for i in 1:tdensity
        for j in 1:tdensity
            txz[tdensity * (i - 1) + j] = 2 * framesz / (tdensity - 1) * (i - (tdensity + 1) / 2)
            tyz[tdensity * (i - 1) + j] = 2 * framesz / (tdensity - 1) * (j - (tdensity + 1) / 2)
            tsz[tdensity * (i - 1) + j] = 100
            tcz[tdensity * (i - 1) + j] = 20 * ((i - 1) / (tdensity - 1)) - 10
            end
        end

    # Initializes values for previous frame tracer coordinates so that the first cycle of
    # Adams-Bashforth works without needing a special case to deal with it
    @sync @distributed for i in 1:tdensity * tdensity
        ptxz[i] = txz[i]
        ptyz[i] = tyz[i]
        end
    end

if (renderTrajectories)
    # Initializes the trajectories array
    for i in 1:tdensity * tdensity
        push!(trajectories, [[txz[i], tyz[i]]])
        end
    end

# Update cycle, each cycle is one frame simulated and rendered
for frames in 1:framesRun
    # Reset c and d so they can be recomputed later
    @sync @distributed for i in 1:(20 * framesz + 1) * (20 * framesz + 1)
        c[i] = 0
        d[i] = 0
        end

    # Set the inital values of the vortices' coordinates in the next frame so the coordinates can
    # be updated properly according to the velocity vectors generated by the other vortices
    @sync @distributed for i in 1:vnum
        nxz[i] = xz[i]
        nyz[i] = yz[i]
        end

    # Set the initial values of the tracers' coordinates in the next frame so the coordinates can
    # be updated properly according to the velocity vectors generated by the vortices
    @sync @distributed for i in 1:tdensity * tdensity
        ntxz[i] = txz[i]
        ntyz[i] = tyz[i]
        end

    # Update positions of virtual vortices
    @sync @distributed for x in 1:vnum
        xz[8 * (x - 1) + 1 + vnum] = -2*framesz + xz[x]
        xz[8 * (x - 1) + 2 + vnum] = -2*framesz + xz[x]
        xz[8 * (x - 1) + 3 + vnum] = -2*framesz + xz[x]
        xz[8 * (x - 1) + 4 + vnum] = xz[x]
        xz[8 * (x - 1) + 5 + vnum] = xz[x]
        xz[8 * (x - 1) + 6 + vnum] = 2*framesz + xz[x]
        xz[8 * (x - 1) + 7 + vnum] = 2*framesz + xz[x]
        xz[8 * (x - 1) + 8 + vnum] = 2*framesz + xz[x]

        yz[8 * (x - 1) + 1 + vnum] = -2*framesz + yz[x]
        yz[8 * (x - 1) + 2 + vnum] = yz[x]
        yz[8 * (x - 1) + 3 + vnum] = 2*framesz + yz[x]
        yz[8 * (x - 1) + 4 + vnum] = 2*framesz + yz[x]
        yz[8 * (x - 1) + 5 + vnum] = -2*framesz + yz[x]
        yz[8 * (x - 1) + 6 + vnum] = 2*framesz + yz[x]
        yz[8 * (x - 1) + 7 + vnum] = yz[x]
        yz[8 * (x - 1) + 8 + vnum] = -2*framesz + yz[x]
        end

    # If we are on the first frame, we re-update the values of the coordinates of the vortices in
    # the previous frame so that Adams-Bashforth runs correctly
    if (frames == 1)
        @sync @distributed for i in 1:9 * vnum
            pxz[i] = xz[i]
            pyz[i] = yz[i]
            end
        end

    # Loop to update all tracers and vortices' coordinates based on the velocities generated by the
    # vortices
    # We loop through all vortices and for each vortex, update the coordinates of all other
    # vortices and tracers and the velocity vectors based on only the velocity generated by the
    # current vortex
    # We can update coordinates since velocities are additive
    for x in 1:9*vnum
        # Update vortices' coordinates
        @sync @distributed for y in 1:vnum
            # Prevent self advection
            if (x == y)
                continue
                end

            # Split update calculations into cases based on whether vortex y's center is within
            # vortex x's radius (the if clause) or not (the else clause) to account for the
            # velocity function's two pieces
            if(rz[x]^2 > (xz[x] - xz[y])^2+(yz[x] - yz[y])^2)
                nxz[y] += (1.5 * vz[x]/rz[x] * (yz[x] - yz[y]) - 0.5 * vz[x]/rz[x] * (pyz[x] - 
                        pyz[y])) * deltatime
                nyz[y] += (1.5 * vz[x]/rz[x] * (xz[y] - xz[x]) - 0.5 * vz[x]/rz[x] * (pxz[y] -
                        pxz[x])) * deltatime
            else
                nxz[y] += (1.5 * (vz[x]*rz[x]/((xz[x] - xz[y])^2 + (yz[x] - yz[y])^2) * (yz[x] - 
                        yz[y])) - 0.5 * (vz[x]*rz[x]/((pxz[x] - pxz[y])^2 + (pyz[x] - pyz[y])^2) *
                                (pyz[x] - pyz[y]))) * deltatime
                nyz[y] += (1.5 * (vz[x]*rz[x]/((xz[x] - xz[y])^2 + (yz[x] - yz[y])^2) * (xz[y] - 
                        xz[x])) - 0.5 * (vz[x]*rz[x]/((pxz[x] - pxz[y])^2 + (pyz[x] - pyz[y])^2) * 
                                (pxz[y] - pxz[x]))) * deltatime
                end
            end

        # Update velocity vectors
        @sync @distributed for i in 1:20*framesz+1
            for j in 1:20*framesz+1
                # Split update calculations into cases based on whether the origin of the velocity
                # vector is within vortex x's radius (the if clause) or not (the else clause) to
                # account for the velocity function's two pieces
                if(rz[x]^2 > (xz[x] - (i-(10*framesz + 1))/10)^2+(yz[x]-(j-(10*framesz + 1))/10)^2)
                    c[(i-1) * (20*framesz + 1) + j] += vz[x]/rz[x] * (yz[x] - (j-(10*framesz + 
                            1))/10)
                    d[(i-1) * (20*framesz + 1) + j] += vz[x]/rz[x] * ((i-(10*framesz + 1))/10 - 
                            xz[x])
                else
                    c[(i-1) * (20*framesz + 1) + j] += vz[x]*rz[x]/((xz[x] - (i - (10*framesz + 
                            1))/10)^2 + (yz[x] - (j - (10*framesz + 1))/10)^2) * (yz[x] - (j-(10*
                                    framesz + 1))/10)
                    d[(i-1) * (20*framesz + 1) + j] += vz[x]*rz[x]/((xz[x] - (i-(10*framesz + 
                            1))/10)^2 + (yz[x] - (j-(10*framesz + 1))/10)^2)*((i-(10*framesz + 
                                    1))/10 - xz[x])
                    end
                end
            end

        # Update tracers' coordinates
        if (renderTracers)
            @sync @distributed for y in 1:(tdensity * tdensity)
                # Split update calculations into cases based on whether the tracer's center is
                # within vortex x's radius (the if clause) or not (the else clause) to account for
                # the velocity function's two pieces
                if(rz[x]^2 > (xz[x] - txz[y])^2+(yz[x] - tyz[y])^2)
                    ntxz[y] += (1.5 * vz[x]/rz[x] * (yz[x] - tyz[y]) - 0.5 * vz[x]/rz[x] * 
                            (pyz[x] - ptyz[y])) * deltatime
                    ntyz[y] += (1.5 * vz[x]/rz[x] * (txz[y] - xz[x]) - 0.5 * vz[x]/rz[x] * 
                            (ptxz[y] - pxz[x])) * deltatime
                else
                    ntxz[y] += (1.5 * (vz[x]*rz[x]/((xz[x] - txz[y])^2 + (yz[x] - tyz[y])^2) * 
                            (yz[x] - tyz[y])) - 0.5 * (vz[x]*rz[x]/((pxz[x] - ptxz[y])^2 + 
                                    (pyz[x] - ptyz[y])^2) * (pyz[x] - ptyz[y]))) * deltatime
                    ntyz[y] += (1.5 * (vz[x]*rz[x]/((xz[x] - txz[y])^2 + (yz[x] - tyz[y])^2) * 
                            (txz[y] - xz[x])) - 0.5 * (vz[x]*rz[x]/((pxz[x] - ptxz[y])^2 + 
                                    (pyz[x] - ptyz[y])^2) * (ptxz[y] - pxz[x]))) * deltatime
                    end
                end
            end
        end

    # Update the vortices' coordinates in the previous frame to prepare for the next frame
    @sync @distributed for i in 1:9*vnum
        pxz[i] = xz[i]
        pyz[i] = yz[i]
        end

    # Update the vortices' current coordinates to the updated coordinates, calculated above
    @sync @distributed for i in 1:vnum
        xz[i] = nxz[i]
        yz[i] = nyz[i]
        end

    # Update the tracers' coordinates in the previous frameto prepare for the next frame, then
    # update the tracers' current coordinates to the updated coordinates, calculated above
    @sync @distributed for i in 1:tdensity * tdensity
        ptxz[i] = txz[i]
        ptyz[i] = tyz[i]
        txz[i] = ntxz[i]
        tyz[i] = ntyz[i]
        end

    # Move vortices that have moved out of the field to appear on the opposite boundary where they
    # moved out of the field
    @sync @distributed for i in 1:vnum
        if (xz[i] > framesz)
            xz[i] -= 2 * framesz
            end
        if (xz[i] < -framesz)
            xz[i] += 2 * framesz
            end
        if (yz[i] > framesz)
            yz[i] -= 2 * framesz
            end
        if (yz[i] < -framesz)
            yz[i] += 2 * framesz
            end
        end

    # Move tracers that have moved out of the field to appear on the opposite boundary where they
    # moved out of the field. If this happens, we also break the trajectory of the tracer apart
    # by pushing an NaN into the trajectory.
    if (renderTracers)
        for i in 1:(tdensity * tdensity)
            if (txz[i] > framesz)
                txz[i] -= 2 * framesz
                if (renderTrajectories)
                    push!(trajectories[i], [NaN, NaN])
                    end
                end
            if (txz[i] < -framesz)
                txz[i] += 2 * framesz
                if (renderTrajectories)
                    push!(trajectories[i], [NaN, NaN])
                    end
                end
            if (tyz[i] > framesz)
                tyz[i] -= 2 * framesz
                if (renderTrajectories)
                    push!(trajectories[i], [NaN, NaN])
                    end
                end
            if (tyz[i] < -framesz)
                tyz[i] += 2 * framesz
                if (renderTrajectories)
                    push!(trajectories[i], [NaN, NaN])
                    end
                end
            end
        end

    # Update tracers' trajectories with latest locations and remove old locations
    for i in 1:tdensity * tdensity
        push!(trajectories[i], [txz[i], tyz[i]])
        while length(trajectories[i]) > trajectoryLength
            popfirst!(trajectories[i])
            end
        end

    # Plot velocity vectors on top of the main plot
    # GRUtils.quiver(a,b,c,d, arrowscale=0.1, headsize=0.001)

    # Plot vortices' locations on top of the main plot
    GRUtils.scatter(xz[1:vnum], yz[1:vnum], rz[1:vnum].*1000, vz[1:vnum] .* 10)

    # Turn holding on so plots are plotted on top of each other and do not overwrite each other
    hold(true)

    # Plot the tracers' locations on top of the main plot
    if (renderTracers)
        # GRUtils.scatter(txz, tyz, tsz, tcz, alpha=0.5)
        end

    # Plot the tracers' trajectories on top of the main plot
    for i in 1:tdensity * tdensity
        for j in 1:length(trajectories[i]) - 1
            if (!isnan(trajectories[i][j][1]) && !isnan(trajectories[i][j+1][1]))
                GRUtils.plot([trajectories[i][j][1], trajectories[i][j + 1][1]],
                             [trajectories[i][j][2], trajectories[i][j + 1][2]],
                             linecolor = GRUtils.color(0,0,0), alpha = 0.9 * j / trajectoryLength + 0.1)
                end
            end
        end

    # Fix the axis ranges of the plot so the autoscale doesn't mess with the size of the plot
    xlim((-framesz, framesz))
    ylim((-framesz, framesz))

    # Set the aspect ratio to 1 to make a square field
    aspectratio(1)

    # Turn holding off so the next cycle of plots in the next frame can be plotted on a blank plot
    hold(false)

    # Disable gridlines in main grid to make plot less messy
    grid(false)

    # Disable colorbar in main grid to make plot less messy
    colorbar(false)

    # Remove axis ticks in main grid to make plot less messy
    mainaxes = mainfig.plots[1].axes
    mainaxes.options[:tickdir] = 0
    mainfig.plots[1].axes = mainaxes

    # Save the generated figures
    # The path to save to is different on Linux than on Windows, so we need to differentiate
    # between the two here
    if (LINUX == true)
        savefig("/home/solver1104/source/Images/Test" * string(frames) * ".png")
    else
        savefig("C:\\Users\\bohan\\OneDrive\\Pictures\\Julia\\Test" * string(frames) * ".png")
        end
    end